#pragma once
#include "game.h"
#include "network.h"
#include <map>

class MCTSNode
{
    friend std::ostream& operator<<(std::ostream& out, MCTSNode const& node);
    MCTSNode* parent_;
    std::map<Move, MCTSNode*> children_;
    Move move_;
    int visits_ = 0;
    float total_val_ = 0;
    float prior_;

public:
    MCTSNode(MCTSNode* node_p, Move mv, float prior_p): parent_(node_p), move_(mv), prior_(prior_p)
    {
    }

    ~MCTSNode();
    void expand(std::vector<std::pair<Move, float>> const& set);
    MCTSNode* cut(Move occurred);
    MCTSNode* select(float c_puct) const;
    Move actByProb(float temp, float move_priors[kBoardSize] = nullptr) const;
    void update(float leaf_value);
    void updateRecursive(float leaf_value);
    void addNoiseToChildPrior(float noise_rate);
    void dump(std::ostream& out, int max_depth, int depth = 0) const;
    float getUpperConfidenceBound(float c_puct) const;

    float getValue() const { return visits_ == 0 ? 0 : total_val_ / visits_; }

    Move getMove() const { return move_; }

    bool isLeaf() const { return children_.size() == 0; }

    bool isRoot() const { return parent_ == nullptr; }
};

std::ostream& operator<<(std::ostream& out, MCTSNode const& node);

class MCTSPurePlayer: public Player
{
    std::string id_;
    int itermax_;
    float c_puct_;
    MCTSNode* root_;

    void swapRoot(MCTSNode* new_root)
    {
        delete root_;
        root_ = new_root;
    }

public:
    MCTSPurePlayer(int itermax, float c_puct);

    ~MCTSPurePlayer() override { delete root_; }

    std::string const& name() const override { return id_; }

    void setItermax(int n);
    void makeId();
    void reset() override;
    Move play(State const& state, ActionMeta& meta) override;
};

class MCTSDeepPlayer: public Player
{
    std::string id_;
    int itermax_;
    float c_puct_;
    MCTSNode* root_;
    std::shared_ptr<FIRNet> net_;

    void swapRoot(MCTSNode* new_root)
    {
        delete root_;
        root_ = new_root;
    }

public:
    MCTSDeepPlayer(std::shared_ptr<FIRNet> nn, int itermax, float c_puct);

    ~MCTSDeepPlayer() override { delete root_; }

    std::string const& name() const override { return id_; }

    void makeId();
    void reset() override;
    Move play(State const& state, ActionMeta& meta) override;
    static void think(int itermax, float c_puct, State const& state, std::shared_ptr<FIRNet> net,
                      MCTSNode* root, bool add_noise_to_root = false);
};
