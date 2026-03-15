#include "llama.h"
#include "pipo_op_perf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama-model.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

static constexpr int PIPO_CPU_BENCH_ITERS = 5;
static constexpr int PIPO_GPU_BENCH_ITERS = 5;
static constexpr int PIPO_CPU_GRAPH_RUNS  = 2;
static constexpr int PIPO_GPU_GRAPH_RUNS  = 5;
static constexpr int PIPO_GPU_MATMUL_RUNS = 5;

#ifdef __EMSCRIPTEN__
#    define N_THREADS 1
#else
#    define N_THREADS 4
#endif

struct bench_result {
	double compute_ms = -1.0;
};

struct backend_bench_result {
	double cpu_time = -1.0;
	double gpu_time = -1.0;
};

struct bench_op_case {
	ggml_op                    op_type     = GGML_OP_NONE;
	ggml_type                  node_type   = GGML_TYPE_F32;
	vector<int64_t>            op_shape;
	vector<uint8_t>            op_param_bytes;
	vector<ggml_type>          src_types;
	vector<vector<int64_t>>    src_nes;
	string                     op_name;
	string                     op_desc;
};

struct consumer_ref {
	const ggml_tensor * node      = nullptr;
	int                 node_idx  = -1;
	int                 src_index = -1;
	bench_op_case       op;
};

struct profile_task {
	string        weight_name;
	size_t        weight_size = 0;
	size_t        other_src_size = 0;
	consumer_ref  consumer;
};

static bench_op_case make_bench_op_case(const ggml_tensor * node) {
	bench_op_case op;
	op.op_type   = node->op;
	op.node_type = node->type;
	op.op_shape.assign(node->ne, node->ne + GGML_MAX_DIMS);
	op.op_param_bytes.resize(sizeof(node->op_params));
	memcpy(op.op_param_bytes.data(), node->op_params, sizeof(node->op_params));
	op.op_name = ggml_op_name(node->op);
	op.op_desc = string(ggml_op_name(node->op)) + ":" + string(ggml_type_name(node->type)) + "(";
	for (int d = 0; d < GGML_MAX_DIMS; ++d) {
		op.op_desc += to_string((long long) node->ne[d]);
		if (d + 1 < GGML_MAX_DIMS) {
			op.op_desc += "x";
		}
	}
	op.op_desc += ")";

	for (int i = 0; i < GGML_MAX_SRC; ++i) {
		const ggml_tensor * src = node->src[i];
		if (src == nullptr) {
			break;
		}
		op.src_types.push_back(src->type);
		op.src_nes.emplace_back(src->ne, src->ne + GGML_MAX_DIMS);
	}

	return op;
}

static string tensor_desc(const ggml_tensor * tensor) {
	string desc = ggml_type_name(tensor->type);
	desc += "(";
	for (int d = 0; d < GGML_MAX_DIMS; ++d) {
		desc += to_string((long long) tensor->ne[d]);
		if (d + 1 < GGML_MAX_DIMS) {
			desc += "x";
		}
	}
	desc += ")";
	return desc;
}

static string make_bench_signature(const bench_op_case & op) {
	string signature;
	signature.reserve(128);
	signature += op.op_name;
	signature += '#';
	signature += ggml_type_name(op.node_type);
	signature += '#';
	for (size_t i = 0; i < op.op_shape.size(); ++i) {
		signature += to_string((long long) op.op_shape[i]);
		if (i + 1 < op.op_shape.size()) {
			signature += 'x';
		}
	}
	return signature;
}

static void init_tensor_uniform(
		ggml_tensor * tensor,
		float         min     = -1.0f,
		float         max     = 1.0f,
		int64_t       int_min = 0,
		int64_t       int_max = 100) {
	size_t nels = ggml_nelements(tensor);

	if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32 ||
		tensor->type == GGML_TYPE_I64) {
		size_t element_size = 0;
		switch (tensor->type) {
			case GGML_TYPE_I8:  element_size = sizeof(int8_t); break;
			case GGML_TYPE_I16: element_size = sizeof(int16_t); break;
			case GGML_TYPE_I32: element_size = sizeof(int32_t); break;
			case GGML_TYPE_I64: element_size = sizeof(int64_t); break;
			default: break;
		}

		vector<uint8_t> data(nels * element_size);

		static const size_t n_threads = N_THREADS > 0 ? N_THREADS : 1;
		static vector<default_random_engine> generators = []() {
			random_device rd;
			vector<default_random_engine> vec;
			vec.reserve(n_threads);
			for (size_t i = 0; i < n_threads; ++i) {
				vec.emplace_back(rd());
			}
			return vec;
		}();

		auto init_thread = [&](size_t ith, size_t start, size_t end) {
			uniform_int_distribution<int64_t> distribution(int_min, int_max);
			auto & gen = generators[ith];

			switch (tensor->type) {
				case GGML_TYPE_I8: {
					int8_t * ptr = reinterpret_cast<int8_t *>(data.data());
					for (size_t i = start; i < end; ++i) {
						ptr[i] = static_cast<int8_t>(distribution(gen));
					}
				} break;
				case GGML_TYPE_I16: {
					int16_t * ptr = reinterpret_cast<int16_t *>(data.data());
					for (size_t i = start; i < end; ++i) {
						ptr[i] = static_cast<int16_t>(distribution(gen));
					}
				} break;
				case GGML_TYPE_I32: {
					int32_t * ptr = reinterpret_cast<int32_t *>(data.data());
					for (size_t i = start; i < end; ++i) {
						ptr[i] = static_cast<int32_t>(distribution(gen));
					}
				} break;
				case GGML_TYPE_I64: {
					int64_t * ptr = reinterpret_cast<int64_t *>(data.data());
					for (size_t i = start; i < end; ++i) {
						ptr[i] = distribution(gen);
					}
				} break;
				default: break;
			}
		};

		if (n_threads == 1) {
			init_thread(0, 0, nels);
		} else {
			vector<future<void>> tasks;
			tasks.reserve(n_threads);
			for (size_t i = 0; i < n_threads; ++i) {
				size_t start = i * nels / n_threads;
				size_t end   = (i + 1) * nels / n_threads;
				tasks.push_back(async(launch::async, init_thread, i, start, end));
			}
			for (auto & task : tasks) {
				task.get();
			}
		}

		ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
		return;
	}

	if (tensor->type == GGML_TYPE_F32) {
		vector<float> data(nels);

		static const size_t n_threads = N_THREADS > 0 ? N_THREADS : 1;
		static vector<default_random_engine> generators = []() {
			random_device rd;
			vector<default_random_engine> vec;
			vec.reserve(n_threads);
			for (size_t i = 0; i < n_threads; ++i) {
				vec.emplace_back(rd());
			}
			return vec;
		}();

		auto init_thread = [&](size_t ith, size_t start, size_t end) {
			uniform_real_distribution<float> distribution(min, max);
			auto & gen = generators[ith];
			for (size_t i = start; i < end; ++i) {
				data[i] = distribution(gen);
			}
		};

		if (n_threads == 1) {
			init_thread(0, 0, nels);
		} else {
			vector<future<void>> tasks;
			tasks.reserve(n_threads);
			for (size_t i = 0; i < n_threads; ++i) {
				size_t start = i * nels / n_threads;
				size_t end   = (i + 1) * nels / n_threads;
				tasks.push_back(async(launch::async, init_thread, i, start, end));
			}
			for (auto & task : tasks) {
				task.get();
			}
		}

		ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
		return;
	}

	if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
		GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

		vector<float> data(nels);
		static const size_t n_threads = N_THREADS > 0 ? N_THREADS : 1;
		static vector<default_random_engine> generators = []() {
			random_device rd;
			vector<default_random_engine> vec;
			vec.reserve(n_threads);
			for (size_t i = 0; i < n_threads; ++i) {
				vec.emplace_back(rd());
			}
			return vec;
		}();

		auto init_thread = [&](size_t ith, size_t start, size_t end) {
			uniform_real_distribution<float> distribution(min, max);
			auto & gen = generators[ith];
			for (size_t i = start; i < end; ++i) {
				data[i] = distribution(gen);
			}
		};

		if (n_threads == 1) {
			init_thread(0, 0, nels);
		} else {
			vector<future<void>> tasks;
			tasks.reserve(n_threads);
			for (size_t i = 0; i < n_threads; ++i) {
				size_t start = i * nels / n_threads;
				size_t end   = (i + 1) * nels / n_threads;
				tasks.push_back(async(launch::async, init_thread, i, start, end));
			}
			for (auto & task : tasks) {
				task.get();
			}
		}

		vector<float> imatrix(tensor->ne[0], 1.0f);
		const float * im = imatrix.data();
		if (!ggml_quantize_requires_imatrix(tensor->type) && data[0] > 0.5f * (min + max)) {
			im = nullptr;
		}

		vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
		const size_t blck_size = ggml_blck_size(tensor->type);
		const size_t n_blocks  = nels / blck_size;

		auto quantize_thread = [&](size_t start, size_t end) {
			ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), start * blck_size, end - start, blck_size, im);
		};

		const size_t n_quant_threads = std::min<size_t>(std::max<size_t>(n_threads / 2, 1), std::max<size_t>(1, n_blocks));
		if (n_quant_threads == 1) {
			quantize_thread(0, n_blocks);
		} else {
			vector<future<void>> tasks;
			tasks.reserve(n_quant_threads);
			for (size_t i = 0; i < n_quant_threads; ++i) {
				size_t start = i * n_blocks / n_quant_threads;
				size_t end   = (i + 1) * n_blocks / n_quant_threads;
				tasks.push_back(async(launch::async, quantize_thread, start, end));
			}
			for (auto & task : tasks) {
				task.get();
			}
		}

		ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
		return;
	}

	GGML_ABORT("Unsupported tensor type in init_tensor_uniform");
}

static double run_single_bench(const bench_op_case & op, ggml_backend_t backend, int n_iter) {
	ggml_init_params init_params = {
		/* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(8192, false),
		/* .mem_base = */ nullptr,
		/* .no_alloc = */ true,
	};

	ggml_context * ctx = ggml_init(init_params);
	if (ctx == nullptr) {
		return -1.0;
	}

	vector<ggml_tensor *> src_tensors(op.src_types.size());
	for (size_t i = 0; i < op.src_types.size(); ++i) {
		src_tensors[i] = ggml_new_tensor(ctx, op.src_types[i], op.src_nes[i].size(), op.src_nes[i].data());
	}

	ggml_tensor * result = ggml_new_tensor(ctx, op.node_type, op.op_shape.size(), op.op_shape.data());
	result->op = op.op_type;

	for (size_t i = 0; i < src_tensors.size(); ++i) {
		result->src[i] = src_tensors[i];
	}
	for (size_t i = src_tensors.size(); i < GGML_MAX_SRC; ++i) {
		result->src[i] = nullptr;
	}

	memcpy(result->op_params, op.op_param_bytes.data(), op.op_param_bytes.size());

	if (!ggml_backend_supports_op(backend, result)) {
		ggml_free(ctx);
		return -1.0;
	}

	ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
	if (!buffer) {
		ggml_free(ctx);
		return -1.0;
	}

	if (op.op_type == GGML_OP_GET_ROWS) {
		init_tensor_uniform(src_tensors.at(0));
		init_tensor_uniform(src_tensors.at(1), 0.0f, 0.0f, 0, src_tensors.at(0)->ne[1] - 1);
	} else {
		for (ggml_tensor * src : src_tensors) {
			init_tensor_uniform(src);
		}
	}

	ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
	ggml_build_forward_expand(gf, result);

	ggml_status status = ggml_backend_graph_compute(backend, gf);
	if (status != GGML_STATUS_SUCCESS) {
		ggml_backend_buffer_free(buffer);
		ggml_free(ctx);
		return -1.0;
	}

	const bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
	int n_runs = is_cpu ? PIPO_CPU_GRAPH_RUNS : PIPO_GPU_GRAPH_RUNS;
	if (op.op_type == GGML_OP_MUL_MAT) {
		n_runs = PIPO_GPU_MATMUL_RUNS;
	}

	for (int i = 1; i < n_runs; ++i) {
		ggml_graph_add_node(gf, result);
	}

	n_iter = max(1, n_iter / n_runs);
	const int64_t t_start = ggml_time_us();
	for (int i = 0; i < n_iter; ++i) {
		if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
			ggml_backend_buffer_free(buffer);
			ggml_free(ctx);
			return -1.0;
		}
	}
	const int64_t t_end = ggml_time_us();

	ggml_backend_buffer_free(buffer);
	ggml_free(ctx);
	return (t_end - t_start) / 1000.0 / (n_iter * n_runs);
}

static double measure_h2d_bandwidth(ggml_backend_t gpu_backend) {
	ggml_init_params init_params = { 1024 * 1024 * 10, nullptr, true };
	ggml_context * ctx = ggml_init(init_params);
	if (ctx == nullptr) {
		return -1.0;
	}

	const size_t tensor_size = 128 * 1024 * 1024;
	ggml_tensor * gpu_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, tensor_size);
	vector<uint8_t> host_data(tensor_size);

	ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu_backend);
	if (!buffer) {
		ggml_free(ctx);
		return -1.0;
	}

	for (int i = 0; i < 5; ++i) {
		ggml_backend_tensor_set(gpu_tensor, host_data.data(), 0, tensor_size);
		ggml_backend_synchronize(gpu_backend);
	}

	double transfer_time_us = 0.0;
	for (int i = 0; i < 5; ++i) {
		const int64_t t_start = ggml_time_us();
		ggml_backend_tensor_set(gpu_tensor, host_data.data(), 0, tensor_size);
		ggml_backend_synchronize(gpu_backend);
		const int64_t t_end = ggml_time_us();
		transfer_time_us += (t_end - t_start);
	}

	ggml_backend_buffer_free(buffer);
	ggml_free(ctx);

	const double transfer_time_ms = transfer_time_us / 5.0 / 1000.0;
	return transfer_time_ms > 0 ? (double) tensor_size / transfer_time_ms : -1.0;
}

static void print_usage(const char * argv0) {
	cerr << "Usage: " << argv0 << " -m <model.gguf> -c <output-path>\n";
}

static string resolve_output_path(const string & raw_output_path) {
	fs::path output_path(raw_output_path);

	if (raw_output_path.empty()) {
		return raw_output_path;
	}

	if (raw_output_path.back() == '/' || raw_output_path.back() == '\\' ||
		(fs::exists(output_path) && fs::is_directory(output_path))) {
		output_path /= "pipo_profile.json";
	}

	fs::path parent = output_path.parent_path();
	if (!parent.empty()) {
		fs::create_directories(parent);
	}

	return output_path.string();
}

static void print_summary(
		const vector<nlohmann::json> & profiles,
		double h2d_bandwidth) {
	cerr << "\n==== Direct Tensor Consumer Summary ====\n";
	for (const auto & profile : profiles) {
		cerr << profile["weight_name"].get<string>()
			 << " size=" << profile["size"].get<size_t>()
			 << " op=" << profile["op_name"].get<string>()
			 << " cpu=" << fixed << setprecision(3) << profile["CPU_time"].get<double>()
			 << " ms gpu=" << fixed << setprecision(3) << profile["GPU_time"].get<double>()
			 << " ms gain=" << fixed << setprecision(6) << profile["gain"].get<double>()
			 << '\n';
	}
	if (h2d_bandwidth > 0.0) {
		cerr << "H2D bandwidth: " << fixed << setprecision(2)
			 << h2d_bandwidth / (1024.0 * 1024.0) << " MiB/ms\n";
	}
}

int main(int argc, char ** argv) {
	string model_path;
	string output_path;
	int n_threads = 4;

	for (int i = 1; i < argc; ++i) {
		const string arg = argv[i];
		if (arg == "-m" && i + 1 < argc) {
			model_path = argv[++i];
		} else if (arg == "-c" && i + 1 < argc) {
			output_path = argv[++i];
		} else if (arg == "-t" && i + 1 < argc) {
			n_threads = std::stoi(argv[++i]);
		}
	}

	if (model_path.empty() || output_path.empty()) {
		print_usage(argv[0]);
		return 1;
	}

	try {
		output_path = resolve_output_path(output_path);
	} catch (const std::exception & e) {
		cerr << "Failed to prepare output path: " << output_path << " error=" << e.what() << '\n';
		return 1;
	}

	ggml_backend_load_all();

	llama_model_params model_params = llama_model_default_params();
	model_params.use_mmap = false;
	model_params.no_alloc = true;

	llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
	if (model == nullptr) {
		cerr << "Failed to load model: " << model_path << '\n';
		return 1;
	}

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx   = 1;
	ctx_params.n_batch = 1;
	ctx_params.no_perf = true;
	ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

	llama_context * ctx = llama_init_from_model(model, ctx_params);
	if (ctx == nullptr) {
		cerr << "Failed to create llama_context\n";
		llama_model_free(model);
		return 1;
	}

	ggml_cgraph * graph = pipo_get_graph(ctx);

	unordered_map<const ggml_tensor *, string> ptr_to_name;
	for (const auto & entry : model->tensors_by_name) {
		ptr_to_name[entry.second] = entry.first;
	}

	vector<profile_task> tasks;
	vector<string> graph_weight_order;
	unordered_map<string, size_t> weight_order_index;

	for (int node_idx = 0; node_idx < ggml_graph_n_nodes(graph); ++node_idx) {
		ggml_tensor * node = ggml_graph_node(graph, node_idx);
		if (node == nullptr || node->op == GGML_OP_NONE || pipo_is_view_op(node->op)) {
			continue;
		}

		bench_op_case op = make_bench_op_case(node);

		for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
			const ggml_tensor * src = node->src[src_index];
			if (src == nullptr) {
				break;
			}
			auto tensor_it = ptr_to_name.find(src);
			if (tensor_it == ptr_to_name.end()) {
				continue;
			}

			const string & weight_name = tensor_it->second;
			if (weight_order_index.find(weight_name) == weight_order_index.end()) {
				weight_order_index[weight_name] = graph_weight_order.size();
				graph_weight_order.push_back(weight_name);
			}

			size_t other_src_size = 0;
			for (int other_src_index = 0; other_src_index < GGML_MAX_SRC; ++other_src_index) {
				const ggml_tensor * other_src = node->src[other_src_index];
				if (other_src == nullptr) {
					break;
				}
				if (other_src_index == src_index) {
					continue;
				}
				other_src_size += ggml_nbytes(other_src);
			}

			tasks.push_back({
				weight_name,
				ggml_nbytes(src),
				other_src_size,
				{ node, node_idx, src_index, op },
			});
		}
	}

	ggml_backend_t cpu_backend = ggml_backend_init_by_name("cpu", nullptr);
	if (cpu_backend == nullptr) {
		cerr << "CPU backend not found\n";
		llama_free(ctx);
		llama_model_free(model);
		return 1;
	}

	ggml_backend_t gpu_backend = nullptr;
	for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
		ggml_backend_dev_t dev = ggml_backend_dev_get(i);
		if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
			gpu_backend = ggml_backend_dev_init(dev, nullptr);
			break;
		}
	}

	if (gpu_backend == nullptr) {
		cerr << "GPU backend not found\n";
		ggml_backend_free(cpu_backend);
		llama_free(ctx);
		llama_model_free(model);
		return 1;
	}

	const double h2d_bandwidth = measure_h2d_bandwidth(gpu_backend);

	vector<nlohmann::json> profiles;
	unordered_map<string, backend_bench_result> bench_cache;
	size_t current_weight_index = 0;
	string current_weight_name;
	for (size_t task_index = 0; task_index < tasks.size(); ++task_index) {
		const profile_task & task = tasks[task_index];
		if (task.weight_name != current_weight_name) {
			current_weight_name = task.weight_name;
			current_weight_index = weight_order_index[task.weight_name] + 1;
			cerr << "[" << current_weight_index << "/" << graph_weight_order.size() << "] profiling weight "
				 << task.weight_name << " size=" << task.weight_size << " bytes\n";
		}

		const consumer_ref & consumer = task.consumer;
			const string bench_signature = make_bench_signature(consumer.op);
			double cpu_time = -1.0;
			double gpu_time = -1.0;
			bool cache_hit = false;

			auto cache_it = bench_cache.find(bench_signature);
			if (cache_it != bench_cache.end()) {
				cpu_time = cache_it->second.cpu_time;
				gpu_time = cache_it->second.gpu_time;
				cache_hit = true;
			} else {
				cpu_time = run_single_bench(consumer.op, cpu_backend, PIPO_CPU_BENCH_ITERS);
				gpu_time = run_single_bench(consumer.op, gpu_backend, PIPO_GPU_BENCH_ITERS);
				bench_cache[bench_signature] = { cpu_time, gpu_time };
			}
			const double gain = (cpu_time >= 0.0 && gpu_time >= 0.0 && task.weight_size > 0)
				? (cpu_time - gpu_time) / (double) task.weight_size
				: -1.0;
			const double transfer_ms = (h2d_bandwidth > 0.0)
				? (double) task.other_src_size / h2d_bandwidth
				: -1.0;

			cerr << "    task " << (task_index + 1) << "/" << tasks.size()
				 << " node=" << consumer.node_idx
				 << " src=" << consumer.src_index
				 << " op=" << consumer.op.op_name
				 << " sig=" << bench_signature
				 << (cache_hit ? " [cache]" : " [bench]")
				 << " cpu=" << fixed << setprecision(3) << cpu_time
				 << " ms gpu=" << fixed << setprecision(3) << gpu_time
				 << " ms gain=" << fixed << setprecision(6) << gain
				 << " other_src_size=" << task.other_src_size
				 << " transfer_ms=" << transfer_ms << '\n';

			profiles.push_back({
				{ "weight_name", task.weight_name },
				{ "size", task.weight_size },
				{ "other_src_size", task.other_src_size },
				{ "transfer_ms", transfer_ms },
				{ "CPU_time", cpu_time },
				{ "GPU_time", gpu_time },
				{ "gain", gain },
				{ "op_name", consumer.op.op_name },
				{ "bench_signature", bench_signature },
				{ "node_index", consumer.node_idx },
				{ "src_index", consumer.src_index },
			});
	}

	nlohmann::json result = nlohmann::json::array();
	for (const auto & profile : profiles) {
		result.push_back(profile);
	}

	{
		ofstream ofs(output_path);
		if (!ofs) {
			cerr << "Failed to open output path: " << output_path << '\n';
			ggml_backend_free(gpu_backend);
			ggml_backend_free(cpu_backend);
			llama_free(ctx);
			llama_model_free(model);
			return 1;
		}
		ofs << result.dump(2) << '\n';
		if (!ofs) {
			cerr << "Failed to write output file: " << output_path << '\n';
			ggml_backend_free(gpu_backend);
			ggml_backend_free(cpu_backend);
			llama_free(ctx);
			llama_model_free(model);
			return 1;
		}
	}

	print_summary(profiles, h2d_bandwidth);
	cerr << "Profile JSON written to: " << output_path << '\n';

	ggml_backend_free(gpu_backend);
	ggml_backend_free(cpu_backend);
	llama_free(ctx);
	llama_model_free(model);
	return 0;
}
