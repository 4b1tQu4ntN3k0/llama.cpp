./scripts/pipo/run_simple.sh release  /home/xiaoguo/lush/models/gguf/Qwen3-14B-Q4_K_M.gguf -r -p 4096 -t 4 base -ngl 21  -ubatch 512 -n 128
./scripts/pipo/run_simple.sh release  /home/xiaoguo/lush/models/gguf/Qwen3-14B-Q4_K_M.gguf -r -p 4096 -t 4 base -ngl 21  -ubatch 512 -n 128 -c examples/pipo-alg/static_cfg/3060/14B.json
./scripts/pipo/run_simple.sh release  /home/xiaoguo/lush/models/gguf/Qwen3-14B-Q4_K_M.gguf -r -p 4096 -t 4 pipo -ubatch 512 -n 128 -c examples/pipo-alg/alg_cfg/perf.json -do 0
./scripts/pipo/run_simple.sh release  /home/xiaoguo/lush/models/gguf/Qwen3-14B-Q4_K_M.gguf -r -p 4096 -t 4 pipo -ubatch 512 -n 128 -c examples/pipo-alg/alg_cfg/perf.json -do 1
