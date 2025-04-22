#include "mcts.h"
#include <iomanip>
#include <sstream>
#include <bshoshany/BS_thread_pool.hpp>

static BS::thread_pool g_thread_pool(kMCTSThreadNum);

MCTSNode::MCTSNode(MCTSNode* parent, Move mv, float prior)
    : parent_(parent), move_(mv), prior_(prior)
{
}

MCTSNode::~MCTSNode()
{
    for (auto const& [_, node]: children_) {
        delete node;
    }
}

MCTSNode::MCTSNode(MCTSNode const& rhs): MCTSNode(nullptr, rhs.move_, rhs.prior_)
{
    visits_ = rhs.visits_;
    total_val_ = rhs.total_val_;
    for (auto const& [_, node]: rhs.children_) {
        auto d = new MCTSNode(*node);
        d->parent_ = this;
        children_[d->move_] = d;
    }
}

void MCTSNode::expand(std::vector<std::pair<Move, float>> const& act_priors)
{
    lock();
    children_.reserve(act_priors.size());
    for (auto& [mv, p]: act_priors) {
        children_[mv] = new MCTSNode(this, mv, p);
    }
    unlock();
}

MCTSNode* MCTSNode::cut(Move occurred)
{
    auto c = children_.find(occurred);
    if (c == children_.end()) {
        MY_THROW("move not found");
    }
    children_.erase(occurred);
    c->second->parent_ = nullptr;
    return c->second;
}

MCTSNode* MCTSNode::select(float c_puct) const
{
    lock();
    MCTSNode* best_node = nullptr;
    float visits_sqrt = std::sqrt(visits_);
    float best_score = -std::numeric_limits<float>::max();
    int best_visits = std::numeric_limits<int>::max();
    for (auto const& [_, node]: children_) {
        node->lock();
        float adjusted_visits = node->visits_ + node->virtual_loss_;
        float adjusted_value =
            (adjusted_visits == 0) ? 0 : (node->total_val_ - node->virtual_loss_) / adjusted_visits;
        float score =
            adjusted_value + (c_puct * node->prior_ * visits_sqrt / (adjusted_visits + 1));
        if (score > best_score) {
            best_node = node;
            best_score = score;
        }
        node->unlock();
    }
    unlock();
    return best_node;
}

Move MCTSNode::actByProb(float temp, float move_priors[kBoardSize]) const
{
    if (kDebugMCTSProb) {
        std::cout << *this << std::endl;
    }
    float move_priors_buffer[kBoardSize] = {0.0f};
    if (move_priors == nullptr) {
        move_priors = move_priors_buffer;
    }
    float max_log_prob = -1 * std::numeric_limits<float>::max();
    float const inv_temp = 1.0 / temp;
    for (auto const& [_, node]: children_) {
        float p = inv_temp * std::log(static_cast<float>(node->visits_) + 1e-10);
        move_priors[node->move_.z()] = p;
        max_log_prob = std::max(max_log_prob, p);
    }
    float sum_prob = 0;
    for (auto const& [_, node]: children_) {
        float p = std::exp(move_priors[node->move_.z()] - max_log_prob);
        move_priors[node->move_.z()] = p;
        sum_prob += p;
    }
    for (auto const& [_, node]: children_) {
        move_priors[node->move_.z()] /= sum_prob;
    }
    std::discrete_distribution<int> discrete(move_priors, move_priors + kBoardSize);
    return Move(discrete(g_random_engine));
}

void MCTSNode::applyVirtualLoss()
{
    lock();
    virtual_loss_ += 1;
    unlock();
}

void MCTSNode::updateAndRevertVirtualLoss(float leaf_value)
{
    lock();
    virtual_loss_ -= 1;
    ++visits_;
    total_val_ += leaf_value;
    unlock();
}

static void genRanDirichlet(const size_t k, float alpha, float theta[])
{
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float sum = 0.0;
    for (size_t i = 0; i < k; i++) {
        theta[i] = gamma(g_random_engine);
        sum += theta[i];
    }
    for (size_t i = 0; i < k; i++) {
        theta[i] /= sum;
    }
}

void MCTSNode::addNoiseToChildPrior(float noise_rate)
{
    static float noise[kBoardSize];
    genRanDirichlet(children_.size(), kDirichletAlpha, noise);
    int i = 0;
    for (auto& [_, node]: children_) {
        node->prior_ = (1 - noise_rate) * node->prior_ + noise_rate * noise[i];
        ++i;
    }
}

void MCTSNode::dump(std::ostream& out, int max_depth, int depth) const
{
    if (depth > max_depth || visits_ == 0) {
        return;
    }
    out << std::string(depth * 4, ' ') << "MCTSNode" << move_ << ": " << std::setw(3)
        << children_.size() << " children, ";
    out << std::setw(6) << std::fixed << std::setprecision(3)
        << (parent_ ? static_cast<float>(visits_) / parent_->visits_ : 1) * 100 << "% / ";
    out << std::setw(3) << visits_ << " visits, " << std::setw(6) << std::fixed
        << std::setprecision(3) << prior_ * 100 << "% prior, " << std::setw(6) << std::fixed
        << std::setprecision(3) << total_val_ / visits_ << " value";
    out << std::endl;
    for (auto const& [_, node]: children_) {
        node->dump(out, max_depth, depth + 1);
    }
}

float MCTSNode::getValue() const { return visits_ == 0 ? 0 : total_val_ / visits_; }

Move MCTSNode::getMove() const { return move_; }

bool MCTSNode::isLeaf() const
{
    lock();
    bool leaf = children_.size() == 0;
    unlock();
    return leaf;
}

bool MCTSNode::isRoot() const { return parent_ == nullptr; }

void MCTSNode::lock() const
{
    while (lck_.test_and_set(std::memory_order_acquire)) {
    }
}

void MCTSNode::unlock() const { lck_.clear(std::memory_order_release); }

std::ostream& operator<<(std::ostream& out, MCTSNode const& node)
{
    node.dump(out, 1);
    return out;
}

MCTSPlayer::MCTSPlayer(int itermax, float c_puct): itermax_(itermax), c_puct_(c_puct)
{
    if (itermax < kMCTSThreadNum || itermax % kMCTSThreadNum != 0) {
        MY_THROW("itermax({}) % kMCTSThreadNum({}) != 0", itermax, kMCTSThreadNum);
    }
    root_ = new MCTSNode;
}

MCTSPlayer::~MCTSPlayer() { delete root_; }

MCTSPlayer::MCTSPlayer(MCTSPlayer const& rhs)
{
    this->itermax_ = rhs.itermax_;
    this->c_puct_ = rhs.c_puct_;
    this->root_ = new MCTSNode(*rhs.root_);
}

float MCTSPlayer::getCpuct() const { return c_puct_; }

int MCTSPlayer::getItermax() const { return itermax_; }

void MCTSPlayer::setItermax(int i) { itermax_ = i; }

void MCTSPlayer::reset()
{
    delete root_;
    root_ = new MCTSNode;
}

void MCTSPlayer::swapRoot(MCTSNode* new_root)
{
    delete root_;
    root_ = new_root;
}

void MCTSPlayer::think(State const& state, int iter_begin, int iter_end)
{
    for (int i = iter_begin; i < iter_end; ++i) {
        State state_copied(state);
        root_->applyVirtualLoss();
        std::vector<MCTSNode*> path{root_};
        MCTSNode* node = root_;
        while (!node->isLeaf()) {
            node = node->select(c_puct_);
            node->applyVirtualLoss();
            path.push_back(node);
            state_copied.next(node->getMove());
        }
        float leaf_value;
        std::vector<std::pair<Move, float>> act_priors;
        eval(state_copied, leaf_value, act_priors);
        if (!act_priors.empty()) {
            node->expand(act_priors);
        }
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            (*it)->updateAndRevertVirtualLoss(leaf_value);
            leaf_value *= -1;
        }
    }
}

Move MCTSPlayer::play(State const& state, ActionMeta& meta)
{
    if (state.getLast().z() != kNoMoveYet && !root_->isLeaf() &&
        state.getLast() != root_->getMove()) {
        swapRoot(root_->cut(state.getLast()));
    }
    if (meta.add_noise_prior) {
        root_->addNoiseToChildPrior(kNoiseRate);
    }
    BS::multi_future<void> loop_future = g_thread_pool.parallelize_loop(
        0, itermax_, [&](int a, int b) { think(state, a, b); }, kMCTSThreadNum);
    loop_future.wait();
    Move act = root_->actByProb(meta.temperature, meta.move_priors);
    swapRoot(root_->cut(act));
    meta.value = root_->getValue();
    return act;
}

MCTSPurePlayer::MCTSPurePlayer(int itermax, float c_puct): MCTSPlayer(itermax, c_puct) {}

MCTSPurePlayer::~MCTSPurePlayer() = default;

std::string MCTSPurePlayer::name() const { return FSTR("i{}u{}", getItermax(), getCpuct()); }

std::shared_ptr<Player> MCTSPurePlayer::clone() const
{
    return std::make_shared<MCTSPurePlayer>(*this);
}

void MCTSPurePlayer::eval(State& state, float& leaf_value,
                          std::vector<std::pair<Move, float>>& act_priors)
{
    Color enemy_side = state.current();
    Color winner = state.getWinner();
    if (!state.over()) {
        float p = 1.0f / state.getOptions().size();
        for (auto const mv: state.getOptions()) {
            act_priors.emplace_back(mv, p);
        }
        winner = state.nextRandTillEnd();
    }
    if (winner == enemy_side) {
        leaf_value = -1.0f;
    } else if (winner == ~enemy_side) {
        leaf_value = 1.0f;
    } else {
        leaf_value = 0.0f;
    }
}

MCTSDeepPlayer::MCTSDeepPlayer(std::shared_ptr<FIRNet> net, int itermax, float c_puct)
    : MCTSPlayer(itermax, c_puct), net_(net)
{
}

MCTSDeepPlayer::~MCTSDeepPlayer() = default;

std::string MCTSDeepPlayer::name() const
{
    return FSTR("i{}u{}@{}", getItermax(), getCpuct(), net_->verno());
}

std::shared_ptr<Player> MCTSDeepPlayer::clone() const
{
    return std::make_shared<MCTSDeepPlayer>(*this);
}

void MCTSDeepPlayer::eval(State& state, float& leaf_value,
                          std::vector<std::pair<Move, float>>& act_priors)
{
    if (!state.over()) {
        net_->evalState(state, &leaf_value, act_priors);
        leaf_value *= -1;
    } else {
        if (state.getWinner() != Color::kEmpty) {
            leaf_value = 1.0f;
        } else {
            leaf_value = 0.0f;
        }
    }
}
