#include "mcts.h"
#include <iomanip>
#include <sstream>

MCTSNode::~MCTSNode()
{
    for (auto const& mn: children_) {
        delete mn.second;
    }
}

void MCTSNode::expand(std::vector<std::pair<Move, float>> const& set)
{
    for (auto& mvp: set) {
        children_[mvp.first] = new MCTSNode(this, mvp.second);
    }
}

MCTSNode* MCTSNode::cut(Move occurred)
{
    auto citer = children_.find(occurred);
    assert(citer != children_.end());
    auto child = citer->second;
    children_.erase(occurred);
    child->parent_ = nullptr;
    return child;
}

std::pair<Move, MCTSNode*> MCTSNode::select(float c_puct) const
{
    std::pair<Move, MCTSNode*> picked(Move(kNoMoveYet), nullptr);
    float max_value = -1 * std::numeric_limits<float>::max();
    for (auto const& mn: children_) {
        float value = mn.second->value(c_puct);
        if (value > max_value) {
            picked = mn;
            max_value = value;
        }
    }
    return picked;
}

Move MCTSNode::actByMostVisted() const
{
    int max_visit = -1 * std::numeric_limits<int>::max();
    Move act(kNoMoveYet);
    if (kDebugMCTSProb) {
        std::cout << "(ROOT): " << *this << std::endl;
    }
    for (auto const& mn: children_) {
        if (kDebugMCTSProb) {
            std::cout << mn.first << ": " << *mn.second << std::endl;
        }
        auto vn = mn.second->visits_;
        if (vn > max_visit) {
            act = mn.first;
            max_visit = vn;
        }
    }
    return act;
}

Move MCTSNode::actByProb(float mcts_move_priors[kBoardSize], float temp) const
{
    float move_priors_buffer[kBoardSize] = {0.0f};
    if (mcts_move_priors == nullptr) {
        mcts_move_priors = move_priors_buffer;
    }
    std::map<int, float> move_priors_map;
    if (kDebugMCTSProb) {
        std::cout << "(ROOT): " << *this << std::endl;
    }
    float alpha = -1 * std::numeric_limits<float>::max();
    for (auto const& mn: children_) {
        if (kDebugMCTSProb) {
            std::cout << mn.first << ": " << *mn.second << std::endl;
        }
        auto vn = mn.second->visits_;
        move_priors_map[mn.first.z()] = 1.0f / temp * std::log(static_cast<float>(vn) + 1e-10);
        if (move_priors_map[mn.first.z()] > alpha) {
            alpha = move_priors_map[mn.first.z()];
        }
    }
    float denominator = 0;
    for (auto& mn: move_priors_map) {
        float value = std::exp(mn.second - alpha);
        move_priors_map[mn.first] = value;
        denominator += value;
    }
    for (auto& mn: move_priors_map) {
        mcts_move_priors[mn.first] = mn.second / denominator;
    }
    float check_sum = 0;
    for (int i = 0; i < kBoardSize; ++i) {
        check_sum += mcts_move_priors[i];
    }
    assert(check_sum > 0.99);
    std::discrete_distribution<int> discrete(mcts_move_priors, mcts_move_priors + kBoardSize);
    return Move(discrete(g_random_engine));
}

void MCTSNode::update(float leaf_value)
{
    ++visits_;
    float delta = (leaf_value - quality_) / static_cast<float>(visits_);
    quality_ += delta;
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
    for (auto& item: children_) {
        item.second->prior_ =
            (1 - noise_rate) * item.second->prior_ + noise_rate * noise_added[prior_cnt];
        ++prior_cnt;
    }
    delete[] noise_added;
}

float MCTSNode::value(float c_puct) const
{
    assert(!isRoot());
    float all_n = static_cast<float>(parent_->visits_);
    float n = visits_ + 1;
    return quality_ + (c_puct * prior_ * std::sqrt(all_n) / n);
}

std::ostream& operator<<(std::ostream& out, MCTSNode const& node)
{
    out << "MCTSNode(" << node.parent_ << "): " << std::setw(3) << node.children_.size()
        << " children, ";
    if (node.parent_ != nullptr) {
        out << std::setw(6) << std::fixed << std::setprecision(3)
            << static_cast<float>(node.visits_) / node.parent_->visits_ * 100 << "% / ";
    }
    out << std::setw(3) << node.visits_ << " visits, " << std::setw(6) << std::fixed
        << std::setprecision(3) << node.prior_ * 100 << "% prior, " << std::setw(6) << std::fixed
        << std::setprecision(3) << node.quality_ << " quality";
    return out;
}

MCTSPurePlayer::MCTSPurePlayer(int itermax, float c_puct): itermax_(itermax), c_puct_(c_puct)
{
    makeId();
    root_ = new MCTSNode(nullptr, 1.0f);
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
    root_ = new MCTSNode(nullptr, 1.0f);
}

Move MCTSPurePlayer::play(State const& state)
{
    if (!(state.getLast().z() == kNoMoveYet) && !root_->isLeaf()) {
        swapRoot(root_->cut(state.getLast()));
    }
    for (int i = 0; i < itermax_; ++i) {
        State state_copied(state);
        MCTSNode* node = root_;
        while (!node->isLeaf()) {
            auto move_node = node->select(c_puct_);
            node = move_node.second;
            state_copied.next(move_node.first);
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
    Move act = root_->actByMostVisted();
    swapRoot(root_->cut(act));
    return act;
}

MCTSDeepPlayer::MCTSDeepPlayer(std::shared_ptr<FIRNet> nn, int itermax, float c_puct)
    : itermax_(itermax), c_puct_(c_puct), net_(nn)
{
    makeId();
    root_ = new MCTSNode(nullptr, 1.0f);
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
    root_ = new MCTSNode(nullptr, 1.0f);
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
            auto move_node = node->select(c_puct);
            node = move_node.second;
            state_copied.next(move_node.first);
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

Move MCTSDeepPlayer::play(State const& state)
{
    if (!(state.getLast().z() == kNoMoveYet) && !root_->isLeaf()) {
        swapRoot(root_->cut(state.getLast()));
    }
    think(itermax_, c_puct_, state, net_, root_);
    Move act = root_->actByProb(nullptr, 1e-3);
    swapRoot(root_->cut(act));
    return act;
}
