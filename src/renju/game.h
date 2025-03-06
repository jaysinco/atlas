#pragma once
#include "utils/logging.h"
#include <random>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

/*
3 * 3 board looks like:
  0 1 2
 ------- Col
0|0 1 2
1|3 4 5
2|6 7 8
 |
Row  => move z(5) = (x(1), y(2))
*/

constexpr int kFiveInRow = 5;
constexpr int kBoardMaxCol = 15;
constexpr int kBoardMaxRow = kBoardMaxCol;
constexpr int kBoardSize = kBoardMaxRow * kBoardMaxCol;
constexpr int kInputFeatureNum = 4;  // self, opponent[[, lastmove], color]
constexpr int kNoMoveYet = -1;
constexpr int kColorOccupySpace = 1;

extern std::mt19937 g_random_engine;

#define ON_BOARD(row, col) (row >= 0 && row < kBoardMaxRow && col >= 0 && col < kBoardMaxCol)

enum class Color
{
    kEmpty,
    kBlack,
    kWhite
};
Color operator~(const Color c);
std::ostream& operator<<(std::ostream& out, Color c);

class Move
{
    int index_;

public:
    explicit Move(int z): index_(z) { assert((z >= 0 && z < kBoardSize) || z == kNoMoveYet); }

    Move(int row, int col)
    {
        assert(ON_BOARD(row, col));
        index_ = row * kBoardMaxCol + col;
    }

    Move(Move const& mv): index_(mv.z()) {}

    int z() const { return index_; }

    int r() const
    {
        assert(index_ >= 0 && index_ < kBoardSize);
        return index_ / kBoardMaxCol;
    }

    int c() const
    {
        assert(index_ >= 0 && index_ < kBoardSize);
        return index_ % kBoardMaxCol;
    }

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
        assert(get(mv) == Color::kEmpty);
        grid_[mv.z()] = c;
    }

    void pushValid(std::vector<Move>& set) const;
    bool winFrom(Move mv) const;
};

std::ostream& operator<<(std::ostream& out, Board const& board);

class State
{
    friend std::ostream& operator<<(std::ostream& out, State const& state);
    Board board_;
    Move last_;
    Color winner_ = Color::kEmpty;
    std::vector<Move> opts_;

public:
    State(): last_(kNoMoveYet) { board_.pushValid(opts_); }

    State(State const& state) = default;

    Move getLast() const { return last_; }

    Color getWinner() const { return winner_; }

    Color current() const;

    bool firstHand() const { return current() == Color::kBlack; }

    void fillFeatureArray(float data[kInputFeatureNum * kBoardSize]) const;

    std::vector<Move> const& getOptions() const
    {
        assert(!over());
        return opts_;
    };

    bool valid(Move mv) const { return std::find(opts_.cbegin(), opts_.cend(), mv) != opts_.end(); }

    bool over() const { return winner_ != Color::kEmpty || opts_.size() == 0; }

    void next(Move mv);
    Color nextRandTillEnd();
};

std::ostream& operator<<(std::ostream& out, State const& state);

struct Player
{
    Player() = default;
    virtual void reset() = 0;
    virtual std::string const& name() const = 0;
    virtual Move play(State const& state) = 0;
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

    Move play(State const& state) override { return state.getOptions()[0]; }

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

    Move play(State const& state) override;
    ~HumanPlayer() override = default;
};