#include "game.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <regex>
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
    Color side = current();
    board_.put(mv, side);
    if (board_.winFrom(mv)) {
        winner_ = side;
    }
    last_ = mv;
    if (auto it = std::find(opts_.begin(), opts_.end(), mv); it != opts_.end()) {
        *it = opts_.back();
        opts_.pop_back();
    } else {
        MY_THROW("invalid move: {}", mv);
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
        auto act = player->play(game, meta);
        game.next(act);
        ++turn;
        if (!silent) {
            std::cout << "\n" << game;
            std::cout << FSTR(
                "{} @ {} win%: {}", ~game.current(), turn,
                meta.value >= -1 ? FSTR("{:.1f}", (1 + meta.value) / 2 * 100) : "n/a");
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
    int p1win = 0, p2win = 0, even = 0;
    Player *temp = nullptr, *pblack = &p1, *pwhite = &p2;
    for (int i = 0; i < round; ++i) {
        temp = pblack, pblack = pwhite, pwhite = temp;
        Player* winner = &play(*pblack, *pwhite);
        if (winner == nullptr) {
            ++even;
        } else if (winner == &p1) {
            ++p1win;
        } else if (winner == &p2) {
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
        std::cout << "win: " << p1.name() << "=" << p1prob * 100 << "%, " << p2.name() << "="
                  << p2prob * 100 << "%, even=" << eprob * 100 << "%, sim=" << round << std::endl;
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
