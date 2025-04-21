#pragma once
#include "game.h"
#include "network.h"
#include <unordered_map>

class MCTSNode
{
    friend std::ostream& operator<<(std::ostream& out, MCTSNode const& node);
    int visits_ = 0;
    float total_val_ = 0;
    float prior_;
    MCTSNode* parent_;
    std::unordered_map<Move, MCTSNode*> children_;
    Move move_;

public:
    MCTSNode(MCTSNode* parent = nullptr, Move mv = Move(kNoMoveYet), float prior = 1.0f)
        : parent_(parent), move_(mv), prior_(prior)
    {
    }

    ~MCTSNode();
    MCTSNode(MCTSNode const& rhs);
    MCTSNode& operator=(MCTSNode const&) = delete;
    void expand(std::vector<std::pair<Move, float>> const& act_priors);
    MCTSNode* cut(Move occurred);
    MCTSNode* select(float c_puct) const;
    Move actByProb(float temp, float move_priors[kBoardSize] = nullptr) const;
    void update(float leaf_value);
    void updateRecursive(float leaf_value);
    void addNoiseToChildPrior(float noise_rate);
    void dump(std::ostream& out, int max_depth, int depth = 0) const;

    float getValue() const { return visits_ == 0 ? 0 : total_val_ / visits_; }

    Move getMove() const { return move_; }

    bool isLeaf() const { return children_.size() == 0; }

    bool isRoot() const { return parent_ == nullptr; }
};

std::ostream& operator<<(std::ostream& out, MCTSNode const& node);

class MCTSPlayer: public Player
{
public:
    MCTSPlayer(int itermax, float c_puct);
    ~MCTSPlayer() override;
    MCTSPlayer(MCTSPlayer const& rhs);
    MCTSPlayer& operator=(MCTSPlayer const&) = delete;
    float getCpuct() const;
    int getItermax() const;
    void setItermax(int i);
    void reset() override;
    Move play(State const& state, ActionMeta& meta) override;

protected:
    virtual void eval(State& state, float& leaf_value,
                      std::vector<std::pair<Move, float>>& act_priors) = 0;

private:
    void swapRoot(MCTSNode* new_root);

    int itermax_;
    float c_puct_;
    MCTSNode* root_;
};

class MCTSPurePlayer: public MCTSPlayer
{
public:
    MCTSPurePlayer(int itermax, float c_puct);
    ~MCTSPurePlayer() override;
    std::string name() const override;
    std::shared_ptr<Player> clone() const override;

protected:
    void eval(State& state, float& leaf_value,
              std::vector<std::pair<Move, float>>& act_priors) override;
};

class MCTSDeepPlayer: public MCTSPlayer
{
public:
    MCTSDeepPlayer(std::shared_ptr<FIRNet> net, int itermax, float c_puct);
    ~MCTSDeepPlayer() override;
    std::string name() const override;
    std::shared_ptr<Player> clone() const override;

protected:
    void eval(State& state, float& leaf_value,
              std::vector<std::pair<Move, float>>& act_priors) override;

private:
    std::shared_ptr<FIRNet> net_;
};
