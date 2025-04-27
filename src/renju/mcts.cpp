#include "mcts.h"
#include <iomanip>
#include <sstream>
#include <stack>

MCTSNode::MCTSNode(MCTSNode* parent, Move mv, float prior)
    : parent_(parent), move_(mv), prior_(prior)
{
}

MCTSNode::~MCTSNode() = default;

MCTSNode::MCTSNode(MCTSNode const& rhs): MCTSNode(nullptr, rhs.move_, rhs.prior_)
{
    visits_ = rhs.visits_;
    total_val_ = rhs.total_val_;
    children_.reserve(rhs.children_.size());
    for (auto const& node: rhs.children_) {
        auto d = new MCTSNode(*node);
        d->parent_ = this;
        children_.push_back(d);
    }
}

void MCTSNode::expand(std::vector<std::pair<Move, float>> const& act_priors)
{
    if (!children_.empty()) {
        return;
    }
    children_.reserve(act_priors.size());
    for (auto& [mv, p]: act_priors) {
        children_.push_back(new MCTSNode(this, mv, p));
    }
}

void MCTSNode::deleteAllChildren()
{
    std::stack<MCTSNode*> nodes_to_delete;
    for (MCTSNode* node: children_) {
        nodes_to_delete.push(node);
    }
    while (!nodes_to_delete.empty()) {
        MCTSNode* current = nodes_to_delete.top();
        nodes_to_delete.pop();
        for (MCTSNode* node: current->children_) {
            nodes_to_delete.push(node);
        }
        delete current;
    }
    children_.clear();
}

MCTSNode* MCTSNode::cut(Move occurred)
{
    auto it = std::find_if(children_.begin(), children_.end(),
                           [=](MCTSNode const* node) { return occurred == node->move_; });
    if (it == children_.end()) {
        MY_THROW("move not found: {}", occurred);
    }
    MCTSNode* c = *it;
    *it = children_.back();
    children_.pop_back();
    c->parent_ = nullptr;
    return c;
}

MCTSNode* MCTSNode::select(float c_puct) const
{
    MCTSNode* best_node = nullptr;
    float visits_sqrt = std::sqrt(visits_);
    float best_score = -std::numeric_limits<float>::max();
    int best_visits = std::numeric_limits<int>::max();
    for (auto const& node: children_) {
        node->lock();
        float adjusted_visits = node->visits_ + node->virtual_loss_;
        float adjusted_value =
            (adjusted_visits == 0) ? 0 : (node->total_val_ - node->virtual_loss_) / adjusted_visits;
        float score =
            adjusted_value + (c_puct * node->prior_ * visits_sqrt / (adjusted_visits + 1));
        if (score > best_score || (score == best_score && node->visits_ < best_visits)) {
            best_node = node;
            best_score = score;
            best_visits = node->visits_;
        }
        node->unlock();
    }
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
    for (auto const& node: children_) {
        float p = inv_temp * std::log(static_cast<float>(node->visits_) + 1e-10);
        move_priors[node->move_.z()] = p;
        max_log_prob = std::max(max_log_prob, p);
    }
    float sum_prob = 0;
    for (auto const& node: children_) {
        float p = std::exp(move_priors[node->move_.z()] - max_log_prob);
        move_priors[node->move_.z()] = p;
        sum_prob += p;
    }
    for (auto const& node: children_) {
        move_priors[node->move_.z()] /= sum_prob;
    }
    std::discrete_distribution<int> discrete(move_priors, move_priors + kBoardSize);
    return Move(discrete(g_random_engine));
}

void MCTSNode::applyVirtualLoss() { virtual_loss_ += 1; }

void MCTSNode::updateAndRevertVirtualLoss(float leaf_value)
{
    virtual_loss_ -= 1;
    ++visits_;
    total_val_ += leaf_value;
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
    for (auto& node: children_) {
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
    for (auto const& node: children_) {
        node->dump(out, max_depth, depth + 1);
    }
}

float MCTSNode::getValue() const { return visits_ == 0 ? 0 : total_val_ / visits_; }

Move MCTSNode::getMove() const { return move_; }

bool MCTSNode::isLeaf() const { return children_.size() == 0; }

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

MCTSPlayer::MCTSPlayer(int itermax, float c_puct, int nthreads)
    : itermax_(itermax), c_puct_(c_puct), nthreads_(nthreads), pool_(nthreads)
{
    root_ = new MCTSNode;
}

MCTSPlayer::~MCTSPlayer()
{
    root_->deleteAllChildren();
    delete root_;
}

MCTSPlayer::MCTSPlayer(MCTSPlayer const& rhs)
{
    this->itermax_ = rhs.itermax_;
    this->c_puct_ = rhs.c_puct_;
    this->root_ = new MCTSNode(*rhs.root_);
}

float MCTSPlayer::getCpuct() const { return c_puct_; }

int MCTSPlayer::getItermax() const { return itermax_; }

int MCTSPlayer::getNumThreads() const { return nthreads_; }

void MCTSPlayer::setItermax(int i) { itermax_ = i; }

void MCTSPlayer::reset()
{
    root_->deleteAllChildren();
    delete root_;
    root_ = new MCTSNode;
}

void MCTSPlayer::swapRoot(MCTSNode* new_root)
{
    root_->deleteAllChildren();
    delete root_;
    root_ = new_root;
}

MyErrCode MCTSPlayer::think(State const& state, int iter_begin, int iter_end)
{
    MY_TRY
    for (int i = iter_begin; i < iter_end; ++i) {
        State state_copied(state);
        std::vector<MCTSNode*> path;
        MCTSNode* node = root_;
        while (true) {
            node->lock();
            node->applyVirtualLoss();
            if (node->isLeaf()) {
                node->unlock();
                break;
            }
            MCTSNode* selected = node->select(c_puct_);
            node->unlock();
            path.push_back(node);
            node = selected;
            state_copied.next(node->getMove());
        }
        float leaf_value;
        std::vector<std::pair<Move, float>> act_priors;
        eval(state_copied, leaf_value, act_priors);
        node->lock();
        if (!act_priors.empty()) {
            node->expand(act_priors);
        }
        node->updateAndRevertVirtualLoss(leaf_value);
        node->unlock();
        leaf_value *= -1;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            MCTSNode* curr = *it;
            curr->lock();
            curr->updateAndRevertVirtualLoss(leaf_value);
            curr->unlock();
            leaf_value *= -1;
        }
    }
    return MyErrCode::kOk;
    MY_CATCH_RET
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
    BS::multi_future<MyErrCode> loop_future = pool_.parallelize_loop(
        0, itermax_, [&](int a, int b) { return think(state, a, b); }, nthreads_);
    std::vector<MyErrCode> res = loop_future.get();
    if (!std::all_of(res.begin(), res.end(),
                     [](MyErrCode code) { return code == MyErrCode::kOk; })) {
        MY_THROW("mcts think failed");
    }
    Move act = root_->actByProb(meta.temperature, meta.move_priors);
    swapRoot(root_->cut(act));
    meta.value = root_->getValue();
    return act;
}

MCTSPurePlayer::MCTSPurePlayer(int itermax, float c_puct, int nthreads)
    : MCTSPlayer(itermax, c_puct, nthreads)
{
}

MCTSPurePlayer::~MCTSPurePlayer() = default;

std::string MCTSPurePlayer::name() const
{
    return FSTR("i{}u{}t{}", getItermax(), getCpuct(), getNumThreads());
}

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

MCTSDeepPlayer::MCTSDeepPlayer(std::shared_ptr<FIRNet> net, int itermax, float c_puct, int nthreads)
    : MCTSPlayer(itermax, c_puct, nthreads), net_(net)
{
    if (itermax < nthreads || itermax % nthreads != 0 || nthreads % 2 != 0) {
        MY_THROW("invalid mcts deep player setup");
    }
}

MCTSDeepPlayer::~MCTSDeepPlayer() = default;

std::string MCTSDeepPlayer::name() const
{
    return FSTR("i{}u{}t{}@{}", getItermax(), getCpuct(), getNumThreads(), net_->verno());
}

std::shared_ptr<Player> MCTSDeepPlayer::clone() const
{
    return std::make_shared<MCTSDeepPlayer>(*this);
}

void MCTSDeepPlayer::eval(State& state, float& leaf_value,
                          std::vector<std::pair<Move, float>>& act_priors)
{
    if (!state.over()) {
        if (auto err = net_->eval(state, &leaf_value, act_priors); err != MyErrCode::kOk) {
            MY_THROW("eval state failed");
        }
        leaf_value *= -1;
    } else {
        if (state.getWinner() != Color::kEmpty) {
            leaf_value = 1.0f;
        } else {
            leaf_value = 0.0f;
        }
    }
}
