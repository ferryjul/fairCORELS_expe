#pragma once

#include "pmap.hh"
#include "alloc.hh"
#include <functional>
#include <queue>
#include <set>
#include <cstdlib>
extern int mode;
extern int random_k_best;
extern int BFS_search_strat;
// pass custom allocator function to track memory allocations in the queue
typedef std::priority_queue<Node*, tracking_vector<Node*, DataStruct::Queue>, 
        std::function<bool(Node*, Node*)> > q;

// orders based on depth (BFS)
static std::function<bool(Node*, Node*)> base_cmp = [](Node* left, Node* right) {
   //return (left->get_num() >= right->get_num());
     if(left->depth() == right->depth()) { //If same depth, fifo
        switch (BFS_search_strat) {
            case 0:
                return (left->depth() >= right->depth());
                break;
            case 1:
                return (left->get_num() >= right->get_num());
                break;
            case 2:
                return (left->objective() >= right->objective());
                break;
            case 3:
                return (left->lower_bound() >= right->lower_bound());
                break;
            case 4:
                if((rand() % 2)) {
                    return false;
                } else {
                    return true;
                }
            default:
                printf("error in BFS search strat!\n");
                return false;
        }
   } else {
       return (left->depth() > right->depth());
   }
};

// orders based on curiosity metric.
static std::function<bool(Node*, Node*)> curious_cmp = [](Node* left, Node* right) {
    return left->get_curiosity() >= right->get_curiosity();
};

// orders based on lower bound.
static std::function<bool(Node*, Node*)> lb_cmp = [](Node* left, Node* right) {
    return left->lower_bound() >= right->lower_bound();
};

// orders based on objective.
static std::function<bool(Node*, Node*)> objective_cmp = [](Node* left, Node* right) {
    return left->objective() >= right->objective();
};

// orders based on depth (DFS)
static std::function<bool(Node*, Node*)> dfs_cmp = [](Node* left, Node* right) {
    return left->depth() <= right->depth();
};

class Queue {
    public:
        Queue(std::function<bool(Node*, Node*)> cmp, char const *type);
        // by default, initialize this as a BFS queue
        Queue() : Queue(base_cmp, "BFS") {};
        Node* front() {
            return q_->top();
        }
        inline void pop() {
            q_->pop();
        }
        void push(Node* node) {
            q_->push(node);
        }
        size_t size() {
            return q_->size();
        }
        bool empty() {
            return q_->empty();
        }
        inline char const * type() {
            return type_;
        }

        std::pair<Node*, tracking_vector<unsigned short, DataStruct::Tree> > select(CacheTree* tree, VECTOR captured) {
            int cnt;
            tracking_vector<unsigned short, DataStruct::Tree> prefix;
            Node *selected_node, *node;
            bool valid = true;
            double lb;
            int nb = 0;
            if(random_k_best > 1) {
                nb = (rand() % random_k_best);
            }
            do {
                selected_node = q_->top();
                q_->pop();
                if(q_->size() < nb) {
                    nb = q_->size();
                }
                if(nb > 0) {
                    int ind = nb;
                    // k nodes + the first popped
                    Node * kBest[nb+1];
                    kBest[nb] = selected_node;
                    // extract the best element k times
                    while(nb > 0) {
                        nb--;
                        kBest[nb] = q_->top();
                        q_->pop();
                    }
                    // keep the kTh extracted as the best
                    selected_node = kBest[0];
                    // push back the k others
                    while(ind > 0) {
                        q_->push(kBest[ind]);
                        ind--;
                    }
                }
                if (tree->ablation() != 2)
                    lb = selected_node->lower_bound() + tree->c();
                else
                    lb = selected_node->lower_bound();
                logger->setCurrentLowerBound(lb);

                node = selected_node;
                // delete leaf nodes that were lazily marked
                if (node->deleted() || (lb >= tree->min_objective())) {
                    tree->decrement_num_nodes();
                    logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
                    delete node;
                    valid = false;
                } else {
                    valid = true;
                }
            } while (!q_->empty() && !valid);
            if (!valid) {
                return std::make_pair((Node*)NULL, prefix);
            }

            rule_vclear(tree->nsamples(), captured);
            while (node != tree->root()) {
                rule_vor(captured,
                         captured, tree->rule(node->id()).truthtable,
                         tree->nsamples(), &cnt);
                prefix.push_back(node->id());
                node = node->parent();
            }
            std::reverse(prefix.begin(), prefix.end());
            return std::make_pair(selected_node, prefix);
        }

    private:
        q* q_;
        char const *type_;
};

extern int bbound(CacheTree* tree, size_t max_num_nodes, Queue* q, PermutationMap* p);

extern void evaluate_children(CacheTree* tree, Node* parent, tracking_vector<unsigned short, DataStruct::Tree> parent_prefix,
        VECTOR parent_not_captured, Queue* q, PermutationMap* p);
