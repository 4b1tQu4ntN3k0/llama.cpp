#include <nlohmann/json.hpp>
#include <ggml-backend.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <regex>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

struct pipo_profile_entry {
	string weight_name;
	size_t size = 0;
	size_t other_src_size = 0;
	double cpu_time = -1.0;
	double gpu_time = -1.0;
	double transfer_ms = 0.0;
	double gain = -1.0;
	string op_name;
	string bench_signature;
	int node_index = -1;
	int src_index = -1;
};

struct parent_choice {
	uint16_t prev_j = 0;
	uint8_t prev_state = 0;
	uint8_t take_gpu = 0;
};

static pipo_profile_entry parse_profile_entry(const nlohmann::json & item) {
	pipo_profile_entry entry;
	entry.weight_name     = item.at("weight_name").get<string>();
	entry.size            = item.at("size").get<size_t>()>>20;
	entry.other_src_size  = item.value("other_src_size", size_t(0)) >> 20;
	entry.cpu_time        = item.at("CPU_time").get<double>();
	entry.gpu_time        = item.at("GPU_time").get<double>();
	entry.transfer_ms     = item.value("transfer_ms", 0.0);
	entry.gain            = item.at("gain").get<double>();
	entry.op_name         = item.value("op_name", string());
	entry.bench_signature = item.value("bench_signature", string());
	entry.node_index      = item.value("node_index", -1);
	entry.src_index       = item.value("src_index", -1);
	return entry;
}

static vector<pipo_profile_entry> load_pipo_profile(const string & json_path) {
	ifstream ifs(json_path);
	if (!ifs) {
		throw runtime_error("failed to open json file: " + json_path);
	}

	nlohmann::json root;
	ifs >> root;
	if (!root.is_array()) {
		throw runtime_error("expected top-level JSON array in: " + json_path);
	}

	vector<pipo_profile_entry> profiles;
	profiles.reserve(root.size());
	for (const auto & item : root) {
		auto p = parse_profile_entry(item);
		if (p.size < 1) continue;
		profiles.push_back(std::move(p));
	}

	return profiles;
}

static void print_usage(const char * argv0) {
	cerr << "Usage: " << argv0 << " <pipo_profile.json> -c <output-dir> [-r <mem-ratio>] [--moe]\n";
}

static int get_gpu_budget_mib(double mem_ratio) {
	ggml_backend_load_all();

	size_t free_bytes = 0;
	size_t total_bytes = 0;
	bool found_gpu = false;

	for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
		ggml_backend_dev_t dev = ggml_backend_dev_get(i);
		if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
			continue;
		}
		ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
		found_gpu = true;
		break;
	}

	if (!found_gpu) {
		throw runtime_error("no GPU backend found");
	}

	if (free_bytes == 0) {
		throw runtime_error("GPU free memory query returned 0 bytes");
	}

	if (!(mem_ratio > 0.0 && mem_ratio <= 1.0)) {
		throw runtime_error("mem ratio must be in (0, 1]");
	}

	const size_t budget_bytes = static_cast<size_t>(free_bytes * mem_ratio);
	const int budget_mib = static_cast<int>(budget_bytes >> 20);
	if (budget_mib <= 0) {
		throw runtime_error("GPU budget in MiB is not positive");
	}

	return budget_mib;
}

static string resolve_output_path(const string & output_dir) {
	fs::path dir(output_dir);
	if (output_dir.empty()) {
		throw runtime_error("output dir is empty");
	}
	fs::create_directories(dir);
	return (dir / "perf.json").string();
}

static bool extract_block_index(const string & weight_name, int & block_index) {
	static const regex block_regex(R"(^blk\.(\d+)\.)");
	smatch match;
	if (!regex_search(weight_name, match, block_regex)) {
		return false;
	}
	block_index = stoi(match[1].str());
	return true;
}

static bool is_forced_cpu_weight(const string & weight_name) {
	return weight_name == "token_embd.weight";
}

static bool is_forced_moe_cpu_weight(const string & weight_name) {
	static const regex moe_exps_regex(R"(ffn_.*_exps)");
	return regex_search(weight_name, moe_exps_regex);
}

int main(int argc, char ** argv) {
	string profile_path;
	string output_dir;
	double mem_ratio = 0.7;
	bool is_moe = false;
	for (int i = 1; i < argc; ++i) {
		const string arg = argv[i];
		if (arg == "-c" && i + 1 < argc) {
			output_dir = argv[++i];
		} else if ((arg == "-r" || arg == "--mem-ratio") && i + 1 < argc) {
			mem_ratio = stod(argv[++i]);
		} else if (arg == "--moe" || arg == "-moe") {
			is_moe = true;
		} else if (profile_path.empty()) {
			profile_path = arg;
		}
	}

	if (profile_path.empty() || output_dir.empty()) {
		print_usage(argv[0]);
		return 1;
	}

	vector<pipo_profile_entry> profiles;
	try {
		profiles = load_pipo_profile(profile_path);
	} catch (const exception & e) {
		cerr << "error: " << e.what() << '\n';
		return 1;
	}

	string output_path;
	try {
		output_path = resolve_output_path(output_dir);
	} catch (const exception & e) {
		cerr << "error: " << e.what() << '\n';
		return 1;
	}

	if (profiles.empty()) {
		nlohmann::json result = {
			{ "offloads", nlohmann::json::array() },
			{ "overrides", nlohmann::json::array() },
		};
		ofstream ofs(output_path);
		if (!ofs) {
			cerr << "error: failed to open output file: " << output_path << '\n';
			return 1;
		}
		ofs << result.dump(2) << '\n';
		cerr << "perf config written to: " << output_path << '\n';
		return 0;
	}

	vector<pipo_profile_entry> dp_profiles;
	dp_profiles.reserve(profiles.size());
	set<string> forced_cpu_names;

	for (const auto & p : profiles) {
		if (is_forced_cpu_weight(p.weight_name)) {
			forced_cpu_names.insert(p.weight_name);
			continue;
		}
		if (is_moe && is_forced_moe_cpu_weight(p.weight_name)) {
			forced_cpu_names.insert(p.weight_name);
			continue;
		}
		dp_profiles.push_back(p);
	}

	const int m_total = get_gpu_budget_mib(mem_ratio);
	const int m = m_total;
	const double neg_inf = -numeric_limits<double>::infinity();
	cerr << "GPU budget: " << m_total << " MiB (" << mem_ratio * 100.0 << "% of current free memory), "
		 << "forced CPU weights: " << forced_cpu_names.size() << ", DP budget: " << m << " MiB\n";

	if (dp_profiles.empty()) {
		vector<string> override_weights;
		override_weights.reserve(profiles.size());
		for (const auto & p : profiles) {
			override_weights.push_back(p.weight_name);
		}

		nlohmann::json offloads = nlohmann::json::array();
		nlohmann::json overrides = nlohmann::json::array();
		for (const string & weight_name : override_weights) {
			int block_index = -1;
			if (extract_block_index(weight_name, block_index) && block_index % 3 == 0 && block_index) {
				offloads.push_back(weight_name);
			}
			overrides.push_back(weight_name);
		}

		nlohmann::json result = {
			{ "offloads", offloads },
			{ "overrides", overrides },
		};

		ofstream ofs(output_path);
		if (!ofs) {
			cerr << "error: failed to open output file: " << output_path << '\n';
			return 1;
		}
		ofs << result.dump(2) << '\n';
		if (!ofs) {
			cerr << "error: failed to write output file: " << output_path << '\n';
			return 1;
		}
		cerr << "perf config written to: " << output_path << '\n';
		cerr << "scores: 0 0\n";
		return 0;
	}

	const int n = dp_profiles.size();

	vector<vector<double>> prev(m + 1, vector<double>(2, neg_inf));
	vector<vector<double>> curr(m + 1, vector<double>(2, neg_inf));
	vector<vector<array<parent_choice, 2>>> parent(n, vector<array<parent_choice, 2>>(m + 1));

	for (int j = 0; j <= m; ++j) {
		prev[j][0] = 0.0;
		parent[0][j][0] = { static_cast<uint16_t>(j), 0, 0 };
		if (j >= (int) dp_profiles[0].size) {
			prev[j][1] = max(0.0, dp_profiles[0].cpu_time - dp_profiles[0].gpu_time - dp_profiles[0].transfer_ms);
			parent[0][j][1] = { static_cast<uint16_t>(j - (int) dp_profiles[0].size), 0, 1 };
		} else {
			parent[0][j][1] = { static_cast<uint16_t>(j), 1, 0 };
		}
	}

	for (int i = 1; i < n; ++i) {
		const double switch_cost = -dp_profiles[i].transfer_ms;
		for (int j = 0; j <= m; ++j) {
			curr[j][0] = prev[j][0];
			curr[j][1] = prev[j][1];
			parent[i][j][0] = { static_cast<uint16_t>(j), 0, 0 };
			parent[i][j][1] = { static_cast<uint16_t>(j), 1, 0 };

			if (prev[j][1] + switch_cost > curr[j][0]) {
				curr[j][0] = prev[j][1] + switch_cost;
				parent[i][j][0] = { static_cast<uint16_t>(j), 1, 0 };
			}

			if (j >= (int) dp_profiles[i].size) {
				const int prev_j = j - (int) dp_profiles[i].size;
				const double gain = dp_profiles[i].cpu_time - dp_profiles[i].gpu_time;
				if (prev[prev_j][1] + gain > curr[j][1]) {
					curr[j][1] = prev[prev_j][1] + gain;
					parent[i][j][1] = { static_cast<uint16_t>(prev_j), 1, 1 };
				}
				if (prev[prev_j][0] + switch_cost + gain > curr[j][1]) {
					curr[j][1] = prev[prev_j][0] + switch_cost + gain;
					parent[i][j][1] = { static_cast<uint16_t>(prev_j), 0, 1 };
				}
			}
		}
		swap(prev, curr);
		for (int j = 0; j <= m; ++j) {
			curr[j][0] = neg_inf;
			curr[j][1] = neg_inf;
		}
	}

	

	int final_state = prev[m][1] > prev[m][0] ? 1 : 0;
	int cur_j = m;
	vector<string> gpu_weights;
	for (int i = n - 1; i >= 0; --i) {
		const parent_choice choice = parent[i][cur_j][final_state];
		if (choice.take_gpu) {
			gpu_weights.push_back(dp_profiles[i].weight_name);
		}
		cur_j = choice.prev_j;
		final_state = choice.prev_state;
	}
	reverse(gpu_weights.begin(), gpu_weights.end());
	set<string> gpu_weight_set(gpu_weights.begin(), gpu_weights.end());

	vector<string> override_weights;
	override_weights.reserve(profiles.size());
	for (const auto & profile : profiles) {
		if (gpu_weight_set.count(profile.weight_name)) {
			continue;
		}
		if (forced_cpu_names.count(profile.weight_name)) {
			override_weights.push_back(profile.weight_name);
			continue;
		}
		override_weights.push_back(profile.weight_name);
	}

	nlohmann::json offloads = nlohmann::json::array();
	nlohmann::json overrides = nlohmann::json::array();
	for (const string & weight_name : override_weights) {
		int block_index = -1;
		if (extract_block_index(weight_name, block_index) && block_index % 3 == 0 && block_index) {
			offloads.push_back(weight_name);
		}
		overrides.push_back(weight_name);
	}

	nlohmann::json result = {
		{ "offloads", offloads },
		{ "overrides", overrides },
	};

	ofstream ofs(output_path);
	if (!ofs) {
		cerr << "error: failed to open output file: " << output_path << '\n';
		return 1;
	}
	ofs << result.dump(2) << '\n';
	if (!ofs) {
		cerr << "error: failed to write output file: " << output_path << '\n';
		return 1;
	}

	cerr << "perf config written to: " << output_path << '\n';
	cerr << "scores: " << prev[m][0] << ' ' << prev[m][1] << '\n';
	return 0;
}
