#include "game.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <map>

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
    }
    assert(opposite != Color::kEmpty);
    return opposite;
}

std::ostream& operator<<(std::ostream& out, Color c)
{
    switch (c) {
        case Color::kEmpty:
            out << ((kColorOccupySpace == 2) ? "  " : " ");
            break;
        case Color::kBlack:
            out << ((kColorOccupySpace == 2) ? "●" : "x");
            break;
        case Color::kWhite:
            out << ((kColorOccupySpace == 2) ? "○" : "o");
            break;
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, Move mv)
{
    return out << "(" << std::setw(2) << mv.r() << ", " << std::setw(2) << mv.c() << ")";
}

void Board::pushValid(std::vector<Move>& set) const
{
    for (int i = 0; i < kBoardSize; ++i) {
        if (get(Move(i)) == Color::kEmpty) {
            set.emplace_back(i);
        }
    }

    std::shuffle(set.begin(), set.end(), g_random_engine);
}

bool Board::winFrom(Move mv) const
{
    if (mv.z() == kNoMoveYet) {
        return false;
    }
    Color side = get(mv);
    assert(side != Color::kEmpty);
    int direct[4][2] = {{0, 1}, {1, 0}, {-1, 1}, {1, 1}};
    int sign[2] = {1, -1};
    for (auto d: direct) {
        int total = 0;
        for (auto s: sign) {
            Move probe = mv;
            while (get(probe) == side) {
                ++total;
                auto r = probe.r() + d[0] * s;
                auto c = probe.c() + d[1] * s;
                if (!ON_BOARD(r, c)) {
                    break;
                }
                probe = Move(r, c);
            }
        }
        if (total - 1 >= kFiveInRow) {
            return true;
        }
    }
    return false;
}

std::ostream& operator<<(std::ostream& out, Board const& board)
{
    if (kColorOccupySpace == 2) {
        out << " ";
    }
    out << "# ";
    for (int c = 0; c < kBoardMaxCol; ++c) {
        out << std::right << std::setw(kColorOccupySpace) << c % 10 << " ";
    }
    out << "\n";
    for (int r = 0; r < kBoardMaxRow; ++r) {
        out << std::right << std::setw(kColorOccupySpace) << r % 10;
        for (int c = 0; c < kBoardMaxCol; ++c) {
            out << "|" << board.get(Move(r, c));
        }
        out << "|\n";
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

void State::fillFeatureArray(float data[kInputFeatureNum * kBoardSize]) const
{
    if (last_.z() == kNoMoveYet) {
        if (kInputFeatureNum > 3) {
            for (int r = 0; r < kBoardMaxRow; ++r) {
                for (int c = 0; c < kBoardMaxCol; ++c) {
                    data[3 * kBoardSize + r * kBoardMaxCol + c] = 1.0f;
                }
            }
        }
        return;
    }
    auto own_side = current();
    auto enemy_side = ~own_side;
    float first = firstHand() ? 1.0f : 0.0f;
    for (int r = 0; r < kBoardMaxRow; ++r) {
        for (int c = 0; c < kBoardMaxCol; ++c) {
            auto side = board_.get(Move(r, c));
            if (side == own_side) {
                data[r * kBoardMaxCol + c] = 1.0f;
            } else if (side == enemy_side) {
                data[kBoardSize + r * kBoardMaxCol + c] = 1.0f;
            }
            if (kInputFeatureNum > 3) {
                data[3 * kBoardSize + r * kBoardMaxCol + c] = first;
            }
        }
    }
    if (kInputFeatureNum > 2) {
        data[2 * kBoardSize + last_.r() * kBoardMaxCol + last_.c()] = 1.0f;
    }
}

void State::next(Move mv)
{
    assert(valid(mv));
    Color side = current();
    board_.put(mv, side);
    if (board_.winFrom(mv)) {
        winner_ = side;
    }
    last_ = mv;
    opts_.erase(std::find(opts_.cbegin(), opts_.cend(), mv));
}

Color State::nextRandTillEnd()
{
    while (!over()) {
        next(opts_[0]);
    }
    return winner_;
}

std::ostream& operator<<(std::ostream& out, State const& state)
{
    if (state.last_.z() == kNoMoveYet) {
        return out << state.board_ << "last move: None";
    } else {
        return out << state.board_ << "last move: " << ~state.current() << state.last_;
    }
}

Player& play(Player& p1, Player& p2, bool silent)
{
    const std::map<Color, Player*> player_color{
        {Color::kBlack, &p1}, {Color::kWhite, &p2}, {Color::kEmpty, nullptr}};
    auto game = State();
    p1.reset();
    p2.reset();
    int turn = 0;
    if (!silent) {
        std::cout << game << std::endl;
    }
    while (!game.over()) {
        auto player = player_color.at(game.current());
        auto act = player->play(game);
        game.next(act);
        ++turn;
        if (!silent) {
            std::cout << game << std::endl;
        }
    }
    auto winner = player_color.at(game.getWinner());
    if (!silent) {
        std::cout << "winner: " << (winner == nullptr ? "no winner, even!" : winner->name())
                  << std::endl;
    }
    return *winner;
}

float benchmark(Player& p1, Player& p2, int round, bool silent)
{
    assert(round > 0);
    int p1win = 0, p2win = 0, even = 0;
    Player *temp = nullptr, *pblack = &p1, *pwhite = &p2;
    for (int i = 0; i < round; ++i) {
        temp = pblack, pblack = pwhite, pwhite = temp;
        Player* winner = &play(*pblack, *pwhite);
        if (winner == nullptr) {
            ++even;
        } else if (winner == &p1) {
            ++p1win;
        } else {
            assert(winner == &p2);
            ++p2win;
        }
        if (!silent) {
            std::cout << std::setfill('0') << "\rscore: total=" << std::setw(4) << i + 1 << ", "
                      << p1.name() << "=" << std::setw(4) << p1win << ", " << p2.name() << "="
                      << std::setw(4) << p2win;
            std::cout.flush();
        }
    }
    if (!silent) {
        std::cout << std::endl;
    }
    float p1prob = static_cast<float>(p1win) / round;
    float p2prob = static_cast<float>(p2win) / round;
    float eprob = static_cast<float>(even) / round;
    if (!silent) {
        std::cout << "benchmark player win probality: " << p1.name() << "=" << p1prob << ", "
                  << p2.name() << "=" << p2prob << ", even=" << eprob << ", sim=" << round
                  << std::endl;
    }
    return p1prob;
}

bool HumanPlayer::getMove(int& row, int& col)
{
    std::string line, srow;
    if (!std::getline(std::cin, line)) {
        return false;
    }
    std::istringstream line_stream(line);
    if (!std::getline(line_stream, srow, ',') || !(line_stream >> col)) {
        return false;
    }
    std::istringstream row_stream(srow);
    if (!(row_stream >> row)) {
        return false;
    }
    return true;
}

Move HumanPlayer::play(State const& state)
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
