#pragma once
#include "config.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <vector>

#define ON_BOARD(row, col) (row >= 0 && row < kBoardMaxRow && col >= 0 && col < kBoardMaxCol)

extern std::mt19937 g_random_engine;

enum class Color
{
    kEmpty,
    kBlack,
    kWhite
};

Color operator~(const Color c);
std::ostream& operator<<(std::ostream& out, Color c);
std::ostream& outputColor(std::ostream& out, Color c, bool checked = false);

class Move
{
    int index_;

public:
    explicit Move(int z): index_(z)
    {
        if ((z < 0 || z >= kBoardSize) && z != kNoMoveYet) {
            MY_THROW("invalid move: {}", z);
        }
    }

    Move(int row, int col)
    {
        if (!ON_BOARD(row, col)) {
            MY_THROW("invalid move: ({}, {})", row, col);
        }
        index_ = row * kBoardMaxCol + col;
    }

    Move(Move const& mv): index_(mv.z()) {}

    int z() const { return index_; }

    int r() const { return index_ / kBoardMaxCol; }

    int c() const { return index_ % kBoardMaxCol; }

    bool operator<(Move const& right) const { return index_ < right.index_; }

    bool operator==(Move const& right) const { return index_ == right.index_; }
};

std::ostream& operator<<(std::ostream& out, Move mv);

class Board
{
    Color grid_[kBoardSize] = {Color::kEmpty};

public:
    Board() = default;

    Color get(Move mv) const { return grid_[mv.z()]; }

    void put(Move mv, Color c)
    {
        if (get(mv) != Color::kEmpty) {
            MY_THROW("invalid move: {}", mv);
        }
        grid_[mv.z()] = c;
    }

    void pushValid(std::vector<Move>& set) const;
    bool winFrom(Move mv) const;
};

std::ostream& operator<<(std::ostream& out, Board const& board);
std::ostream& outputBoard(std::ostream& out, Board const& board, Move last = Move(kNoMoveYet));

class State
{
    friend std::ostream& operator<<(std::ostream& out, State const& state);
    Board board_;
    Move last_;
    Color winner_ = Color::kEmpty;
    std::vector<Move> opts_;

public:
    State(): last_(kNoMoveYet)
    {
        board_.pushValid(opts_);
        shuffleOptions();
    }

    State(State const& state) = default;

    Move getLast() const { return last_; }

    Color getWinner() const { return winner_; }

    Color current() const;

    bool firstHand() const { return current() == Color::kBlack; }

    void fillFeatureArray(float data[kInputFeatureNum * kBoardSize]) const;

    std::vector<Move> const& getOptions() const { return opts_; };

    bool valid(Move mv) const { return std::find(opts_.cbegin(), opts_.cend(), mv) != opts_.end(); }

    bool over() const { return winner_ != Color::kEmpty || opts_.size() == 0; }

    void shuffleOptions() { std::shuffle(opts_.begin(), opts_.end(), g_random_engine); }

    void next(Move mv);
    Color nextRandTillEnd();
};

std::ostream& operator<<(std::ostream& out, State const& state);

struct ActionMeta
{
    float p_mov = -1;
    float p_win = -1;
};

struct Player
{
    Player() = default;
    virtual void reset() = 0;
    virtual std::string const& name() const = 0;
    virtual Move play(State const& state, ActionMeta& meta) = 0;
    virtual ~Player() = default;
};

Player& play(Player& p1, Player& p2, bool silent = true);
float benchmark(Player& p1, Player& p2, int round, bool silent = true);

class RandomPlayer: public Player
{
    std::string id_;

public:
    explicit RandomPlayer(std::string const& name): id_(name) {}

    void reset() override {}

    std::string const& name() const override { return id_; }

    Move play(State const& state, ActionMeta& meta) override { return state.getOptions().back(); }

    ~RandomPlayer() override = default;
};

class HumanPlayer: public Player
{
    std::string id_;
    static bool getMove(int& row, int& col);

public:
    explicit HumanPlayer(std::string const& name): id_(name) {}

    void reset() override {}

    std::string const& name() const override { return id_; }

    Move play(State const& state, ActionMeta& meta) override;
    ~HumanPlayer() override = default;
};