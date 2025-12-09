[] 测试计算图简化成单层，假设GPU放了一些层，CPU放了剩余层，我想构建一个GPU层和一个CPU层，执行的时候这两个层绑定的tensor实时切换。这里需要一个切换的方法，加入到推理的代码中，当然推理的代码最好重写，后续还有更多修改。

[] 拆分llama.cpp定义好的模型，拆解其计算图为静态 GPU 层、动态 GPU 层，CPU 层三部分。其中静态GPU层的层数是可变的，动态GPU层有两层，CPU层有一层。其中CPU层和动态GPU层是不断重用的（切换绑定的tensor），进一步完善推理的代码。

[] 新增数据传输方法，使用 ggml底层提供的能力，实时将CPU上的层加载到GPU的tensor上，加载完后再调用切换的方法。这里和第二步不同的是，GPU上不再需要存储原本分配给他的层，而是只存储GPU固定层，剩余层全部交由数据传输和切换来动态加载和推理。此时需要进一步完善推理代码，支持上述功能。

[] 设计算法完成系统的调度过程，来替代之前步骤中的手动固定参数（如GPU固定层的层数和overlap时CPU执行层数）的方法。



技术方案：基于手动 I/O 的动态权重加载
目标：在显存/内存受限场景下，绕过 OS 的 mmap 自动换页机制，实现高效、可控的“磁盘 -> CPU -> GPU”数据流。

核心思想：

元数据持久化：加载模型时，不加载动态层的权重数据，仅记录其在磁盘文件中的位置（Offset & Size）。
文件句柄常驻：保持模型文件句柄打开，避免重复 open/close 开销。
手动 I/O (pread)：在推理过程中，使用 pread 直接从文件读取特定层的数据到预分配的 Pinned Memory 缓冲区。
流水线传输：读取完成后，立即通过 PCIe (H2D) 传输到 GPU，实现 I/O 与计算的重叠。
具体实现路径：

数据结构扩展 (llama_model)：

新增 struct LayerDiskInfo：存储 Tensor 的 offset, size, file_index 等信息。
新增 layer_disk_map：vector<vector<LayerDiskInfo>>，按层索引存储所有 Tensor 的磁盘位置。
新增 file_handles：保持模型文件的 FILE* 或 fd。
加载逻辑修改 (load_tensors)：

对于标记为 Dynamic 的层：
跳过内存分配：不调用 ggml_backend_alloc_ctx_tensors。
跳过数据读取：不从 loader 读取权重。
记录元数据：将 loader 中的 Tensor 偏移量信息保存到 layer_disk_map。
创建空壳 Tensor：仅保留 Tensor 的 Shape 和 Type 信息，data 指针设为 NULL。
推理逻辑修改 (llama_decode)：

在计算第 N 层之前（或并行地）：
分配缓冲区：从预分配的 CPU 缓冲池中获取一个 Buffer。
执行读取：遍历 layer_disk_map[N]，使用 pread 将数据读入 Buffer。
绑定 Tensor：将空壳 Tensor 的 data 指针指向该 Buffer（或直接传输到 GPU Buffer）。
触发传输：调用 ggml_backend_tensor_set 上传数据。