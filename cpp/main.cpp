/**
 * Marvin Chess AI - C++ ONNX Runtime Inference Demo
 * 
 * Demonstrates running the Marvin ONNX model with CUDA acceleration.
 */

#include <iostream>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

#include <onnxruntime_cxx_api.h>

#include "encoding.hpp"

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "=== Marvin Chess AI - C++ ONNX Runtime Demo ===" << std::endl;
    
    // -------------------------------------------------------------------------
    // 1. Initialize ONNX Runtime with CUDA
    // -------------------------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "marvin");
    Ort::SessionOptions session_options;
    
    // Enable CUDA provider
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = 0;  // No limit
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;
    
    try {
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[OK] CUDA execution provider added" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[WARN] Failed to add CUDA provider: " << e.what() << std::endl;
        std::cerr << "       Falling back to CPU..." << std::endl;
    }
    
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // -------------------------------------------------------------------------
    // 2. Load Model
    // -------------------------------------------------------------------------
    const wchar_t* model_path = L"marvin_small.onnx";
    
    std::cout << "Loading model: marvin_small.onnx" << std::endl;
    std::cout.flush();
    
    Ort::Session* session_ptr = nullptr;
    try {
        session_ptr = new Ort::Session(env, model_path, session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }
    Ort::Session& session = *session_ptr;
    
    // Print provider info
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();
    
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "  Inputs: " << num_inputs << std::endl;
    std::cout << "  Outputs: " << num_outputs << std::endl;
    std::cout.flush();
    
    // -------------------------------------------------------------------------
    // 3. Prepare Input Tensors
    // -------------------------------------------------------------------------
    constexpr int64_t batch_size = 1;
    
    // Check if we should load Python-generated inputs
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // 3. Prepare Input Tensors (matching Python defaults from encoding.py)
    // -------------------------------------------------------------------------
    
    // Board history: (1, 8, 64) int64
    // Initialize with 0. Python's encoding.py pads history with [0]*64.
    // We only fill the first slot (current position) for the start of a game.
    std::vector<int64_t> board_history(batch_size * marvin::HISTORY_LEN * marvin::NUM_SQUARES, 0);
    auto starting_board = marvin::get_starting_board();
    std::copy(starting_board.begin(), starting_board.end(), board_history.begin());
    
    // Time history: (1, 8) float32 - default is 0.0 (no time history)
    // Python: time_history_s = [0.0] * HISTORY_LEN, then normalized by /60.0
    std::vector<float> time_history(batch_size * marvin::HISTORY_LEN, 0.0f);
    
    // Rep flags: (1, 8) float32 - no repetitions
    std::vector<float> rep_flags(batch_size * marvin::HISTORY_LEN, 0.0f);
    
    // Castling: (1, 4) float32 - all rights available
    std::vector<float> castling = {1.0f, 1.0f, 1.0f, 1.0f};
    
    // EP mask: (1, 64) float32 - no en passant
    std::vector<float> ep_mask(batch_size * marvin::NUM_SQUARES, 0.0f);
    
    // Scalars: (1, 8) float32 - using Python ContextOptions defaults
    // Python defaults: active_elo=1900, opponent_elo=1900, active_clock_s=300.0,
    //                  opponent_clock_s=300.0, active_inc_s=0.0, opponent_inc_s=0.0,
    //                  halfmove_clock=0
    marvin::ContextOptions ctx;  // Uses same defaults as Python
    auto scalars_arr = marvin::compute_scalars(ctx);
    std::vector<float> scalars(scalars_arr.begin(), scalars_arr.end());
    
    // TC category: (1,) int64 - computed from base + 40*inc = 300 + 0 = 300 < 600 => Blitz
    std::vector<int64_t> tc_cat = {marvin::TC_BLITZ};
    
    // Legal mask: (1, 4098) bool - all 20 legal starting moves
    // Indices from python-chess: from_square*64 + to_square
    std::vector<bool> legal_mask(batch_size * marvin::NUM_POLICY_OUTPUTS, false);
    // Knight moves
    legal_mask[407] = true;  // g1h3
    legal_mask[405] = true;  // g1f3
    legal_mask[82] = true;   // b1c3
    legal_mask[80] = true;   // b1a3
    // Pawn single pushes (rank 2 -> rank 3)
    legal_mask[983] = true;  // h2h3
    legal_mask[918] = true;  // g2g3
    legal_mask[853] = true;  // f2f3
    legal_mask[788] = true;  // e2e3
    legal_mask[723] = true;  // d2d3
    legal_mask[658] = true;  // c2c3
    legal_mask[593] = true;  // b2b3
    legal_mask[528] = true;  // a2a3
    // Pawn double pushes (rank 2 -> rank 4)
    legal_mask[991] = true;  // h2h4
    legal_mask[926] = true;  // g2g4
    legal_mask[861] = true;  // f2f4
    legal_mask[796] = true;  // e2e4
    legal_mask[731] = true;  // d2d4
    legal_mask[666] = true;  // c2c4
    legal_mask[601] = true;  // b2b4
    legal_mask[536] = true;  // a2a4
    
    // -------------------------------------------------------------------------
    // 4. Create ONNX Tensors
    // -------------------------------------------------------------------------
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    std::array<int64_t, 3> board_history_shape = {batch_size, marvin::HISTORY_LEN, marvin::NUM_SQUARES};
    std::array<int64_t, 2> time_history_shape = {batch_size, marvin::HISTORY_LEN};
    std::array<int64_t, 2> rep_flags_shape = {batch_size, marvin::HISTORY_LEN};
    std::array<int64_t, 2> castling_shape = {batch_size, 4};
    std::array<int64_t, 2> ep_mask_shape = {batch_size, marvin::NUM_SQUARES};
    std::array<int64_t, 2> scalars_shape = {batch_size, 8};
    std::array<int64_t, 1> tc_cat_shape = {batch_size};
    std::array<int64_t, 2> legal_mask_shape = {batch_size, marvin::NUM_POLICY_OUTPUTS};
    
    auto board_history_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, board_history.data(), board_history.size(),
        board_history_shape.data(), board_history_shape.size());
    
    auto time_history_tensor = Ort::Value::CreateTensor<float>(
        memory_info, time_history.data(), time_history.size(),
        time_history_shape.data(), time_history_shape.size());
    
    auto rep_flags_tensor = Ort::Value::CreateTensor<float>(
        memory_info, rep_flags.data(), rep_flags.size(),
        rep_flags_shape.data(), rep_flags_shape.size());
    
    auto castling_tensor = Ort::Value::CreateTensor<float>(
        memory_info, castling.data(), castling.size(),
        castling_shape.data(), castling_shape.size());
    
    auto ep_mask_tensor = Ort::Value::CreateTensor<float>(
        memory_info, ep_mask.data(), ep_mask.size(),
        ep_mask_shape.data(), ep_mask_shape.size());
    
    auto scalars_tensor = Ort::Value::CreateTensor<float>(
        memory_info, scalars.data(), scalars.size(),
        scalars_shape.data(), scalars_shape.size());
    
    auto tc_cat_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, tc_cat.data(), tc_cat.size(),
        tc_cat_shape.data(), tc_cat_shape.size());
    
    // For bool tensor, ONNX Runtime expects a special handling
    // Convert bool to uint8 for the tensor
    std::vector<uint8_t> legal_mask_u8(legal_mask.begin(), legal_mask.end());
    
    auto legal_mask_tensor = Ort::Value::CreateTensor(
        memory_info, legal_mask_u8.data(), legal_mask_u8.size(),
        legal_mask_shape.data(), legal_mask_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
    
    // -------------------------------------------------------------------------
    // 5. Run Inference
    // -------------------------------------------------------------------------
    const char* input_names[] = {
        "board_history", "time_history", "rep_flags", "castling",
        "ep_mask", "scalars", "tc_cat", "legal_mask"
    };
    const char* output_names[] = {
        "move_logits", "value_out", "value_cls_out",
        "value_error_out", "time_cls_out", "start_square_logits"
    };
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(board_history_tensor));
    input_tensors.push_back(std::move(time_history_tensor));
    input_tensors.push_back(std::move(rep_flags_tensor));
    input_tensors.push_back(std::move(castling_tensor));
    input_tensors.push_back(std::move(ep_mask_tensor));
    input_tensors.push_back(std::move(scalars_tensor));
    input_tensors.push_back(std::move(tc_cat_tensor));
    input_tensors.push_back(std::move(legal_mask_tensor));
    
    std::cout << "\nRunning inference..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), input_tensors.size(),
        output_names, 6
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "Inference time: " << duration_ms << " ms" << std::endl;
    
    // -------------------------------------------------------------------------
    // 6. Process Outputs
    // -------------------------------------------------------------------------
    // Get move logits
    float* move_logits_data = output_tensors[0].GetTensorMutableData<float>();
    
    // Apply mask and find top moves
    std::vector<std::pair<float, int>> scored_moves;
    for (int i = 0; i < marvin::NUM_POLICY_OUTPUTS - 2; ++i) {  // Exclude resign/flag
        if (legal_mask[i]) {
            scored_moves.push_back({move_logits_data[i], i});
        }
    }
    
    // Sort by score descending
    std::sort(scored_moves.begin(), scored_moves.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "\n--- Top Moves (from starting position) ---" << std::endl;
    int top_n = std::min(5, static_cast<int>(scored_moves.size()));
    for (int i = 0; i < top_n; ++i) {
        auto [from_sq, to_sq] = marvin::decode_move_index(scored_moves[i].second);
        std::cout << "  " << (i + 1) << ". " 
                  << marvin::move_to_uci(from_sq, to_sq)
                  << " (score: " << scored_moves[i].first << ")" << std::endl;
    }
    
    // Get value output
    float* value_data = output_tensors[1].GetTensorMutableData<float>();
    std::cout << "\nValue: " << value_data[0] << std::endl;
    
    // Get WDL
    float* wdl_data = output_tensors[2].GetTensorMutableData<float>();
    std::cout << "WDL logits: [L=" << wdl_data[0] 
              << ", D=" << wdl_data[1] 
              << ", W=" << wdl_data[2] << "]" << std::endl;
    
    // -------------------------------------------------------------------------
    // 7. Benchmark (optional)
    // -------------------------------------------------------------------------
    std::cout << "\n--- Benchmarking (100 iterations) ---" << std::endl;
    
    constexpr int NUM_ITERS = 100;
    
    // Recreate tensors for benchmark (they were moved)
    board_history_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, board_history.data(), board_history.size(),
        board_history_shape.data(), board_history_shape.size());
    time_history_tensor = Ort::Value::CreateTensor<float>(
        memory_info, time_history.data(), time_history.size(),
        time_history_shape.data(), time_history_shape.size());
    rep_flags_tensor = Ort::Value::CreateTensor<float>(
        memory_info, rep_flags.data(), rep_flags.size(),
        rep_flags_shape.data(), rep_flags_shape.size());
    castling_tensor = Ort::Value::CreateTensor<float>(
        memory_info, castling.data(), castling.size(),
        castling_shape.data(), castling_shape.size());
    ep_mask_tensor = Ort::Value::CreateTensor<float>(
        memory_info, ep_mask.data(), ep_mask.size(),
        ep_mask_shape.data(), ep_mask_shape.size());
    scalars_tensor = Ort::Value::CreateTensor<float>(
        memory_info, scalars.data(), scalars.size(),
        scalars_shape.data(), scalars_shape.size());
    tc_cat_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, tc_cat.data(), tc_cat.size(),
        tc_cat_shape.data(), tc_cat_shape.size());
    legal_mask_tensor = Ort::Value::CreateTensor(
        memory_info, legal_mask_u8.data(), legal_mask_u8.size(),
        legal_mask_shape.data(), legal_mask_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
    
    input_tensors.clear();
    input_tensors.push_back(std::move(board_history_tensor));
    input_tensors.push_back(std::move(time_history_tensor));
    input_tensors.push_back(std::move(rep_flags_tensor));
    input_tensors.push_back(std::move(castling_tensor));
    input_tensors.push_back(std::move(ep_mask_tensor));
    input_tensors.push_back(std::move(scalars_tensor));
    input_tensors.push_back(std::move(tc_cat_tensor));
    input_tensors.push_back(std::move(legal_mask_tensor));
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        session.Run(Ort::RunOptions{nullptr},
                   input_names, input_tensors.data(), input_tensors.size(),
                   output_names, 6);
    }
    
    // Timed run
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
        session.Run(Ort::RunOptions{nullptr},
                   input_names, input_tensors.data(), input_tensors.size(),
                   output_names, 6);
    }
    end = std::chrono::high_resolution_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / NUM_ITERS;
    double positions_per_sec = 1000.0 / avg_ms;
    
    std::cout << "Average latency: " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << positions_per_sec << " positions/sec" << std::endl;
    
    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
