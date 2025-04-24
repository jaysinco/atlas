#pragma once
#include "game.h"
#include "network.h"
#include <atomic>
#include <bshoshany/BS_thread_pool.hpp>

class MCTSNode
{
public:
    MCTSNode(MCTSNode* parent = nullptr, Move mv = Move(kNoMoveYet), float prior = 1.0f);
    MCTSNode(MCTSNode const& rhs);
    ~MCTSNode();
    MCTSNode& operator=(MCTSNode const&) = delete;
    void expand(std::vector<std::pair<Move, float>> const& act_priors);
    void deleteAllChildren();
    MCTSNode* cut(Move occurred);
    MCTSNode* select(float c_puct) const;
    Move actByProb(float temp, float move_priors[kBoardSize] = nullptr) const;
    void applyVirtualLoss();
    void updateAndRevertVirtualLoss(float leaf_value);
    void addNoiseToChildPrior(float noise_rate);
    void dump(std::ostream& out, int max_depth, int depth = 0) const;
    float getValue() const;
    Move getMove() const;
    bool isLeaf() const;
    bool isRoot() const;
    void lock() const;
    void unlock() const;

private:
    friend std::ostream& operator<<(std::ostream& out, MCTSNode const& node);
    int visits_ = 0;
    float total_val_ = 0;
    float prior_;
    mutable std::atomic_flag lck_ = ATOMIC_FLAG_INIT;
    MCTSNode* parent_;
    std::vector<MCTSNode*> children_;
    float virtual_loss_ = 0;
    Move const move_;
};

std::ostream& operator<<(std::ostream& out, MCTSNode const& node);

class MCTSPlayer: public Player
{
public:
    MCTSPlayer(int itermax, float c_puct, int nthreads);
    ~MCTSPlayer() override;
    MCTSPlayer(MCTSPlayer const& rhs);
    MCTSPlayer& operator=(MCTSPlayer const&) = delete;
    float getCpuct() const;
    int getItermax() const;
    int getNumThreads() const;
    void setItermax(int i);
    void reset() override;
    Move play(State const& state, ActionMeta& meta) override;

protected:
    virtual void eval(State& state, float& leaf_value,
                      std::vector<std::pair<Move, float>>& act_priors) = 0;

private:
    MyErrCode think(State const& state, int iter_begin, int iter_end);
    void swapRoot(MCTSNode* new_root);

    int itermax_;
    float c_puct_;
    int nthreads_;
    BS::thread_pool pool_;
    MCTSNode* root_;
};

class MCTSPurePlayer: public MCTSPlayer
{
public:
    MCTSPurePlayer(int itermax, float c_puct, int nthreads);
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
    MCTSDeepPlayer(std::shared_ptr<FIRNet> net, int itermax, float c_puct, int nthreads);
    ~MCTSDeepPlayer() override;
    std::string name() const override;
    std::shared_ptr<Player> clone() const override;

protected:
    void eval(State& state, float& leaf_value,
              std::vector<std::pair<Move, float>>& act_priors) override;

private:
    std::shared_ptr<FIRNet> net_;
};
