#include "mcts.h"
#include <iomanip>
#include <sstream>

MCTSNode::~MCTSNode()
{
    for (auto const& [_, node]: children_) {
        delete node;
    }
}

void MCTSNode::expand(std::vector<std::pair<Move, float>> const& set)
{
    for (auto& [mv, p]: set) {
        children_[mv] = new MCTSNode(this, mv, p);
    }
}

MCTSNode* MCTSNode::cut(Move occurred)
{
    auto citer = children_.find(occurred);
    if (citer == children_.end()) {
        MY_THROW("move not found");
    }
    auto child = citer->second;
    children_.erase(occurred);
    child->parent_ = nullptr;
    return child;
}

MCTSNode* MCTSNode::select(float c_puct) const
{
    MCTSNode* n = nullptr;
    float u_max = -std::numeric_limits<float>::max();
    for (auto const& [_, node]: children_) {
        float u = node->getUpperConfidenceBound(c_puct);
        if (u > u_max) {
            n = node;
            u_max = u;
        }
    }
    return n;
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

void MCTSNode::update(float leaf_value)
{
    ++visits_;
    total_val_ += leaf_value;
}

void MCTSNode::updateRecursive(float leaf_value)
{
    if (parent_ != nullptr) {
        parent_->updateRecursive(-1 * leaf_value);
    }
    update(leaf_value);
}

static void genRanDirichlet(const size_t k, float alpha, float theta[])
{
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float norm = 0.0;
    for (size_t i = 0; i < k; i++) {
        theta[i] = gamma(g_random_engine);
        norm += theta[i];
    }
    for (size_t i = 0; i < k; i++) {
        theta[i] /= norm;
    }
}

void MCTSNode::addNoiseToChildPrior(float noise_rate)
{
    auto noise_added = new float[children_.size()];
    genRanDirichlet(children_.size(), kDirichletAlpha, noise_added);
    int prior_cnt = 0;
    for (auto& [_, node]: children_) {
        node->prior_ = (1 - noise_rate) * node->prior_ + noise_rate * noise_added[prior_cnt];
        ++prior_cnt;
    }
    delete[] noise_added;
}

float MCTSNode::getUpperConfidenceBound(float c_puct) const
{
    if (isRoot()) {
        MY_THROW("root node have no value");
    }
    float all_visits = static_cast<float>(parent_->visits_);
    return getValue() + (c_puct * prior_ * std::sqrt(all_visits) / (visits_ + 1));
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

std::ostream& operator<<(std::ostream& out, MCTSNode const& node)
{
    node.dump(out, 1);
    return out;
}

MCTSPurePlayer::MCTSPurePlayer(int itermax, float c_puct): itermax_(itermax), c_puct_(c_puct)
{
    makeId();
    root_ = new MCTSNode(nullptr, Move(kNoMoveYet), 1.0f);
}

void MCTSPurePlayer::makeId()
{
    std::ostringstream ids;
    ids << "mcts" << itermax_;
    id_ = ids.str();
}

void MCTSPurePlayer::setItermax(int n)
{
    itermax_ = n;
    makeId();
}

void MCTSPurePlayer::reset()
{
    delete root_;
    root_ = new MCTSNode(nullptr, Move(kNoMoveYet), 1.0f);
}

Move MCTSPurePlayer::play(State const& state, ActionMeta& meta)
{
    if (!(state.getLast().z() == kNoMoveYet) && !root_->isLeaf()) {
        swapRoot(root_->cut(state.getLast()));
    }
    for (int i = 0; i < itermax_; ++i) {
        State state_copied(state);
        state_copied.shuffleOptions();
        MCTSNode* node = root_;
        while (!node->isLeaf()) {
            node = node->select(c_puct_);
            state_copied.next(node->getMove());
        }
        Color enemy_side = state_copied.current();
        Color winner = state_copied.getWinner();
        if (!state_copied.over()) {
            int n_options = state_copied.getOptions().size();
            std::vector<std::pair<Move, float>> move_priors;
            for (auto const mv: state_copied.getOptions()) {
                move_priors.emplace_back(mv, 1.0f / n_options);
            }
            node->expand(move_priors);
            winner = state_copied.nextRandTillEnd();
        }
        float leaf_value;
        if (winner == enemy_side) {
            leaf_value = -1.0f;
        } else if (winner == ~enemy_side) {
            leaf_value = 1.0f;
        } else {
            leaf_value = 0.0f;
        }
        node->updateRecursive(leaf_value);
    }
    float move_priors[kBoardSize] = {0.0f};
    Move act = root_->actByProb(1e-3, move_priors);
    meta.p_mov = move_priors[act.z()];
    swapRoot(root_->cut(act));
    meta.p_win = root_->getValue();
    return act;
}

MCTSDeepPlayer::MCTSDeepPlayer(std::shared_ptr<FIRNet> nn, int itermax, float c_puct)
    : itermax_(itermax), c_puct_(c_puct), net_(nn)
{
    makeId();
    root_ = new MCTSNode(nullptr, Move(kNoMoveYet), 1.0f);
}

void MCTSDeepPlayer::makeId()
{
    std::ostringstream ids;
    ids << "mcts" << itermax_ << "_net" << net_->verno();
    id_ = ids.str();
}

void MCTSDeepPlayer::reset()
{
    delete root_;
    root_ = new MCTSNode(nullptr, Move(kNoMoveYet), 1.0f);
}

void MCTSDeepPlayer::think(int itermax, float c_puct, State const& state,
                           std::shared_ptr<FIRNet> net, MCTSNode* root, bool add_noise_to_root)
{
    if (add_noise_to_root) {
        root->addNoiseToChildPrior(kNoiseRate);
    }
    for (int i = 0; i < itermax; ++i) {
        State state_copied(state);
        MCTSNode* node = root;
        while (!node->isLeaf()) {
            node = node->select(c_puct);
            state_copied.next(node->getMove());
        }
        float leaf_value;
        if (!state_copied.over()) {
            std::vector<std::pair<Move, float>> net_move_priors;
            net->evalState(state_copied, &leaf_value, net_move_priors);
            node->expand(net_move_priors);
            leaf_value *= -1;
        } else {
            if (state_copied.getWinner() != Color::kEmpty) {
                leaf_value = 1.0f;
            } else {
                leaf_value = 0.0f;
            }
        }
        node->updateRecursive(leaf_value);
    }
}

Move MCTSDeepPlayer::play(State const& state, ActionMeta& meta)
{
    if (!(state.getLast().z() == kNoMoveYet) && !root_->isLeaf()) {
        swapRoot(root_->cut(state.getLast()));
    }
    think(itermax_, c_puct_, state, net_, root_);
    float move_priors[kBoardSize] = {0.0f};
    Move act = root_->actByProb(1e-3, move_priors);
    meta.p_mov = move_priors[act.z()];
    swapRoot(root_->cut(act));
    meta.p_win = root_->getValue();
    return act;
}
