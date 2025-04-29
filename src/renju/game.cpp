#include "game.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <regex>
#include <map>
#include <boost/timer/timer.hpp>

Color operator~(const Color c)
{
    Color opposite;
    switch (c) {
        case Color::kBlack:
            opposite = Color::kWhite;
            break;
        case Color::kWhite:
            opposite = Color::kBlack;
            break;
        case Color::kEmpty:
            opposite = Color::kEmpty;
            break;
        default:
            MY_THROW("invalid color: {}", static_cast<int>(c));
    }
    return opposite;
}

std::ostream& operator<<(std::ostream& out, Color c) { return outputColor(out, c); }

std::ostream& outputColor(std::ostream& out, Color c, bool checked)
{
    switch (c) {
        case Color::kEmpty:
            out << (kBoardRichSymbol ? "·" : "·");
            break;
        case Color::kBlack:
            out << (kBoardRichSymbol ? (checked ? "" : "") : (checked ? "X" : "x"));
            break;
        case Color::kWhite:
            out << (kBoardRichSymbol ? (checked ? "" : "") : (checked ? "O" : "o"));
            break;
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, Move mv)
{
    return out << "(" << std::setw(2) << mv.r() << ", " << std::setw(2) << mv.c() << ")";
}

void Board::pushValid(std::vector<Move>& opts) const
{
    for (int i = 0; i < kBoardSize; ++i) {
        if (get(Move(i)) == Color::kEmpty) {
            opts.emplace_back(i);
        }
    }
}

bool Board::winFrom(Move mv) const
{
    if (mv.z() == kNoMoveYet) {
        return false;
    }
    Color side = get(mv);
    if (side == Color::kEmpty) {
        return false;
    }
    int constexpr kDirections[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (auto const& dir: kDirections) {
        int count = 1;
        for (int sign = -1; sign <= 1; sign += 2) {
            int r = mv.r() + sign * dir[0];
            int c = mv.c() + sign * dir[1];
            while (ON_BOARD(r, c) && get(Move(r, c)) == side) {
                if (++count >= kFiveInRow) {
                    return true;
                }
                r += sign * dir[0];
                c += sign * dir[1];
            }
        }
    }
    return false;
}

bool Board::operator==(Board const& right) const
{
    for (int i = 0; i < kBoardSize; ++i) {
        Move mv(i);
        if (get(mv) != right.get(mv)) {
            return false;
        }
    }
    return true;
}

std::ostream& operator<<(std::ostream& out, Board const& board) { return outputBoard(out, board); }

std::ostream& outputBoard(std::ostream& out, Board const& board, Move last)
{
    static char const* symtbl[16] = {"0", "1", "2", "3", "4", "5", "6", "7",
                                     "8", "9", "a", "b", "c", "d", "e", "#"};
    out << symtbl[15] << " ";
    for (int c = 0; c < kBoardMaxCol; ++c) {
        out << std::right << symtbl[c] << " ";
    }
    out << "\n";
    for (int r = 0; r < kBoardMaxRow; ++r) {
        out << std::right << symtbl[r];
        for (int c = 0; c < kBoardMaxCol; ++c) {
            out << " ";
            Move mv(r, c);
            outputColor(out, board.get(mv), mv == last);
        }
        out << " \n";
    }
    return out;
}

Color State::current() const
{
    if (last_.z() == kNoMoveYet) {
        return Color::kBlack;
    }
    return ~board_.get(last_);
}

State::State(float const data[kInputFeatureNum * kBoardSize])
{
    if (kInputFeatureNum < 4) {
        MY_THROW("feature number must >= 4");
    }
    auto own_side = Color::kWhite;
    if (data[3 * kBoardSize + 0] > 0) {
        own_side = Color::kBlack;
    }
    auto enemy_side = ~own_side;
    for (int row = 0; row < kBoardMaxRow; ++row) {
        for (int col = 0; col < kBoardMaxCol; ++col) {
            Move mv(row, col);
            if (data[2 * kBoardSize + row * kBoardMaxCol + col] > 0) {
                last_ = mv;
            }
            Color side = Color::kEmpty;
            if (data[row * kBoardMaxCol + col] > 0) {
                side = own_side;
            } else if (data[kBoardSize + row * kBoardMaxCol + col] > 0) {
                side = enemy_side;
            } else {
                opts_.push_back(mv);
            }
            if (side != Color::kEmpty) {
                board_.put(mv, side);
                if (board_.winFrom(mv)) {
                    winner_ = side;
                }
            }
        }
    }
    if (winner_ != Color::kEmpty) {
        opts_.clear();
    }
}

void State::fillFeatureArray(float data[kInputFeatureNum * kBoardSize]) const
{
    if (last_.z() == kNoMoveYet) {
        if (kInputFeatureNum > 3) {
            std::fill_n(data + 3 * kBoardSize, kBoardSize, 1.0f);
        }
        return;
    }
    auto own_side = current();
    auto enemy_side = ~own_side;
    float first = firstHand() ? 1.0f : 0.0f;
    for (int i = 0; i < kBoardSize; ++i) {
        const Color side = board_.get(Move(i));
        if (side == own_side) {
            data[i] = 1.0f;
        } else if (side == enemy_side) {
            data[kBoardSize + i] = 1.0f;
        }
        if (kInputFeatureNum > 3) {
            data[3 * kBoardSize + i] = first;
        }
    }
    if (kInputFeatureNum > 2) {
        data[2 * kBoardSize + last_.z()] = 1.0f;
    }
}

void State::next(Move mv)
{
    if (auto it = std::find(opts_.begin(), opts_.end(), mv); it != opts_.end()) {
        *it = opts_.back();
        opts_.pop_back();
    } else {
        MY_THROW("invalid move: {}", mv);
    }
    Color side = current();
    board_.put(mv, side);
    last_ = mv;
    if (board_.winFrom(mv)) {
        winner_ = side;
        opts_.clear();
    }
}

Color State::nextRandTillEnd()
{
    std::shuffle(opts_.begin(), opts_.end(), g_random_engine);
    while (!over()) {
        next(opts_.front());
    }
    return winner_;
}

bool State::operator==(State const& right) const
{
    if (board_ != right.board_) {
        return false;
    }
    if (last_ != right.last_) {
        return false;
    }
    if (winner_ != right.winner_) {
        return false;
    }
    if (opts_.size() != right.opts_.size()) {
        return false;
    }
    for (int i = 0; i < opts_.size(); ++i) {
        if (std::find(right.opts_.begin(), right.opts_.end(), opts_[i]) == right.opts_.end()) {
            return false;
        }
    }
    return true;
}

std::ostream& operator<<(std::ostream& out, State const& state)
{
    return outputBoard(out, state.board_, state.last_);
}

Player& play(Player& p1, Player& p2, bool silent)
{
    const std::map<Color, Player*> player_color{
        {Color::kBlack, &p1}, {Color::kWhite, &p2}, {Color::kEmpty, nullptr}};
    State game;
    p1.reset();
    p2.reset();
    int turn = 0;
    if (!silent) {
        std::cout << game;
    }
    while (!game.over()) {
        auto player = player_color.at(game.current());
        ActionMeta meta;
        boost::timer::cpu_timer timer;
        auto act = player->play(game, meta);
        timer.stop();
        game.next(act);
        ++turn;
        if (!silent) {
            std::cout << "\n" << game;
            std::cout << FSTR(
                "{} {} @ {:.3f}s {}", ~game.current(), turn, timer.elapsed().wall * 1e-9,
                meta.value >= -1 ? FSTR("{:.1f}%", (1 + meta.value) / 2 * 100) : "n/a");
            std::cout << std::endl;
        }
    }
    auto winner = player_color.at(game.getWinner());
    if (!silent) {
        std::cout << "\nwinner: " << (winner == nullptr ? "no winner, even!" : winner->name())
                  << std::endl;
    }
    return *winner;
}

float benchmark(Player& p1, Player& p2, int round, bool silent)
{
    if (round <= 0) {
        return 0.0f;
    }
    int p1win = 0, p1win_white = 0, p2win = 0, p2win_white = 0, even = 0;
    Player *pblack = &p1, *pwhite = &p2;
    std::cout << FSTR("\rwin: sim=0 even=0 {}=0|0 {}=0|0", p1.name(), p2.name()) << std::flush;
    for (int i = 0; i < round; ++i) {
        Player* winner = &play(*pblack, *pwhite);
        int winner_is_white = winner == pwhite ? 1 : 0;
        if (winner == nullptr) {
            ++even;
        } else if (winner == &p1) {
            ++p1win;
            p1win_white += winner_is_white;
        } else if (winner == &p2) {
            ++p2win;
            p2win_white += winner_is_white;
        }
        if (!silent) {
            std::cout << FSTR("\rwin: sim={} even={} {}={}|{} {}={}|{}", i + 1, even, p1.name(),
                              p1win, p1win_white, p2.name(), p2win, p2win_white)
                      << std::flush;
        }
        std::swap(pblack, pwhite);
    }
    if (!silent) {
        std::cout << std::endl;
    }
    float p1prob = static_cast<float>(p1win) / round;
    float p2prob = static_cast<float>(p2win) / round;
    float eprob = static_cast<float>(even) / round;
    if (!silent) {
        std::cout << FSTR("win: sim={} even={:.0f}% {}={:.0f}% {}={:.0f}%", round, eprob * 100,
                          p1.name(), p1prob * 100, p2.name(), p2prob * 100)
                  << std::endl;
    }
    return p1prob;
}

bool HumanPlayer::getMove(int& row, int& col)
{
    static std::regex pattern(R"(^([0-9a-eA-E])(?:[, \t]+)([0-9a-eA-E])$)");
    std::string line;
    if (!std::getline(std::cin, line)) {
        return false;
    }
    std::smatch matches;
    if (!std::regex_match(line, matches, pattern)) {
        return false;
    }
    row = std::stoi(matches[1].str(), nullptr, 16);
    col = std::stoi(matches[2].str(), nullptr, 16);
    return true;
}

Move HumanPlayer::play(State const& state, ActionMeta& meta)
{
    int col, row;
    while (true) {
        std::cout << state.current() << "(" << id_ << "): ";
        std::cout.flush();
        if (getMove(row, col)) {
            auto mv = Move(row, col);
            if (state.valid(mv)) {
                return mv;
            }
        }
    }
}
