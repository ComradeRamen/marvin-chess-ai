#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace marvin {

// =============================================================================
// Constants (must match Python encoding.py)
// =============================================================================
constexpr int HISTORY_LEN = 8;
constexpr int NUM_SQUARES = 64;
constexpr int NUM_POLICY_OUTPUTS = 4098;

// Piece encoding values
enum Piece : int64_t {
    EMPTY = 0,
    W_PAWN = 1, W_KNIGHT = 2, W_BISHOP = 3, W_ROOK = 4, W_QUEEN = 5, W_KING = 6,
    B_PAWN = 7, B_KNIGHT = 8, B_BISHOP = 9, B_ROOK = 10, B_QUEEN = 11, B_KING = 12
};

// Time control categories
enum TimeControlCategory : int64_t {
    TC_BLITZ = 0,
    TC_RAPID = 1,
    TC_CLASSICAL = 2
};

// Known increment values (seconds)
constexpr std::array<float, 5> KNOWN_INCREMENTS = {0.0f, 2.0f, 3.0f, 5.0f, 10.0f};

// =============================================================================
// Utility Functions
// =============================================================================

inline float clamp_to_known_increment(float inc_s) {
    float best = KNOWN_INCREMENTS[0];
    float best_diff = std::abs(inc_s - best);
    for (size_t i = 1; i < KNOWN_INCREMENTS.size(); ++i) {
        float diff = std::abs(inc_s - KNOWN_INCREMENTS[i]);
        if (diff < best_diff) {
            best = KNOWN_INCREMENTS[i];
            best_diff = diff;
        }
    }
    return best;
}

inline TimeControlCategory get_tc_category(float base_seconds, float inc_seconds) {
    float duration = base_seconds + 40.0f * inc_seconds;
    if (duration < 600.0f) return TC_BLITZ;
    if (duration < 1800.0f) return TC_RAPID;
    return TC_CLASSICAL;
}

// =============================================================================
// Scalar Normalization (matches Python encoding.py scalars)
// =============================================================================

struct ContextOptions {
    int active_elo = 1900;
    int opponent_elo = 1900;
    float active_clock_s = 300.0f;   // Python default
    float opponent_clock_s = 300.0f; // Python default
    float active_inc_s = 0.0f;
    float opponent_inc_s = 0.0f;
    float tc_base_s = 180.0f;  // Match compare_onnx_pytorch.py (used for TC category)
    int halfmove_clock = 0;
    int fullmove_number = 1;
    bool is_white_turn = true;
};

inline std::array<float, 8> compute_scalars(const ContextOptions& ctx) {
    float active_elo_norm = static_cast<float>(ctx.active_elo - 1900) / 700.0f;
    float opp_elo_norm = static_cast<float>(ctx.opponent_elo - 1900) / 700.0f;
    
    int ply = ctx.fullmove_number * 2 - (ctx.is_white_turn ? 0 : 1);
    float ply_norm = static_cast<float>(ply) / 100.0f;
    
    float active_clock_norm = std::log1p(std::max(0.0f, ctx.active_clock_s)) / 10.0f;
    float opp_clock_norm = std::log1p(std::max(0.0f, ctx.opponent_clock_s)) / 10.0f;
    
    float clamped_active_inc = clamp_to_known_increment(ctx.active_inc_s);
    float clamped_opp_inc = clamp_to_known_increment(ctx.opponent_inc_s);
    float active_inc_norm = clamped_active_inc / 30.0f;
    float opp_inc_norm = clamped_opp_inc / 30.0f;
    
    float hmc_norm = static_cast<float>(ctx.halfmove_clock) / 100.0f;
    
    return {
        active_elo_norm,
        opp_elo_norm,
        ply_norm,
        active_clock_norm,
        opp_clock_norm,
        active_inc_norm,
        opp_inc_norm,
        hmc_norm
    };
}

// =============================================================================
// Starting Position Board (pre-encoded for demo)
// =============================================================================

// Standard chess starting position encoded per encoding.md
// Squares 0-63: a1, b1, ..., h1, a2, ..., h8
inline std::array<int64_t, 64> get_starting_board() {
    std::array<int64_t, 64> board{};
    board.fill(EMPTY);
    
    // Rank 1 (White pieces): a1-h1 = indices 0-7
    board[0] = W_ROOK;   board[1] = W_KNIGHT; board[2] = W_BISHOP; board[3] = W_QUEEN;
    board[4] = W_KING;   board[5] = W_BISHOP; board[6] = W_KNIGHT; board[7] = W_ROOK;
    
    // Rank 2 (White pawns): a2-h2 = indices 8-15
    for (int i = 8; i < 16; ++i) board[i] = W_PAWN;
    
    // Rank 7 (Black pawns): a7-h7 = indices 48-55
    for (int i = 48; i < 56; ++i) board[i] = B_PAWN;
    
    // Rank 8 (Black pieces): a8-h8 = indices 56-63
    board[56] = B_ROOK;  board[57] = B_KNIGHT; board[58] = B_BISHOP; board[59] = B_QUEEN;
    board[60] = B_KING;  board[61] = B_BISHOP; board[62] = B_KNIGHT; board[63] = B_ROOK;
    
    return board;
}

// =============================================================================
// Move Index Decoding
// =============================================================================

inline std::pair<int, int> decode_move_index(int idx) {
    // idx = from_sq * 64 + to_sq
    int from_sq = idx / 64;
    int to_sq = idx % 64;
    return {from_sq, to_sq};
}

inline std::string square_name(int sq) {
    char file = 'a' + (sq % 8);
    char rank = '1' + (sq / 8);
    return std::string{file, rank};
}

inline std::string move_to_uci(int from_sq, int to_sq) {
    return square_name(from_sq) + square_name(to_sq);
}

}  // namespace marvin
