#include "utils.hh"
#include <stdio.h>
#include <assert.h>
#include <sys/utsname.h>
#include <math.h>

Logger::Logger(double c, size_t nrules, int verbosity, char* log_fname, int freq) {
      _c = c;
      _nrules = nrules - 1;
      _v = verbosity;
      _freq = freq;
      setLogFileName(log_fname);
      initPrefixVec();
}

/*
 * Sets the logger file name and writes the header line to the file.
 */
void Logger::setLogFileName(char *fname) {
    if (_v < 1) return;

    printf("writing logs to: %s\n\n", fname);
    _f.open(fname, ios::out | ios::trunc);

    _f << "total_time,evaluate_children_time,node_select_time,"
       << "rule_evaluation_time,lower_bound_time,lower_bound_num,"
       << "objective_time,objective_num,"
       << "tree_insertion_time,tree_insertion_num,queue_insertion_time,evaluate_children_num,"
       << "permutation_map_insertion_time,permutation_map_insertion_num,permutation_map_memory,"
       << "current_lower_bound,tree_min_objective,tree_prefix_length,"
       << "tree_num_nodes,tree_num_evaluated,tree_memory,"
       << "queue_size,queue_min_length,queue_memory,"
       << "pmap_size,pmap_null_num,pmap_discard_num,"
       << "log_remaining_space_size,prefix_lengths" << endl;
}

/*
 * Writes current stats about the execution to the log file.
 */
void Logger::dumpState() {
    if (_v < 1) return;

    // update timestamp here
    setTotalTime(time_diff(_state.initial_time));

    _f << _state.total_time << ","
       << _state.evaluate_children_time << ","
       << _state.node_select_time << ","
       << _state.rule_evaluation_time << ","
       << _state.lower_bound_time << ","
       << _state.lower_bound_num << ","
       << _state.objective_time << ","
       << _state.objective_num << ","
       << _state.tree_insertion_time << ","
       << _state.tree_insertion_num << ","
       << _state.queue_insertion_time << ","
       << _state.evaluate_children_num << ","
       << _state.permutation_map_insertion_time << ","
       << _state.permutation_map_insertion_num << ","
       << _state.pmap_memory << ","
       << _state.current_lower_bound << ","
       << _state.tree_min_objective << ","
       << _state.tree_prefix_length << ","
       << _state.tree_num_nodes << ","
       << _state.tree_num_evaluated << ","
       << _state.tree_memory << ","
       << _state.queue_size << ","
       << _state.queue_min_length << ","
       << _state.queue_memory << ","
       << _state.pmap_size << ","
       << _state.pmap_null_num << ","
       << _state.pmap_discard_num << ","
       << getLogRemainingSpaceSize() << ","
       << dumpPrefixLens().c_str() << endl;
}

/*
 * Uses GMP library to dump a string version of the remaining state space size.
 * This number is typically very large (e.g. 10^20) which is why we use GMP instead of a long.
 * Note: this function may not work on some Linux machines.
 */
std::string Logger::dumpRemainingSpaceSize() {
    mpz_class s(_state.remaining_space_size);
    return s.get_str();
}

/*
 * Function to convert vector of remaining prefix lengths to a string format for logging.
 */
std::string Logger::dumpPrefixLens() {
    std::string s = "";
    for(size_t i = 0; i < _nrules; ++i) {
        if (_state.prefix_lens[i] > 0) {
            s += std::to_string(i);
            s += ":";
            s += std::to_string(_state.prefix_lens[i]);
            s += ";";
        }
    }
    return s;
}

/*
 * Given a rulelist and predictions, will output a human-interpretable form to a file.
 */
void print_final_rulelist(const tracking_vector<unsigned short, DataStruct::Tree>& rulelist,
                          const tracking_vector<bool, DataStruct::Tree>& preds,
                          const bool latex_out,
                          const rule_t rules[],
                          const rule_t labels[],
                          char fname[]) {
    assert(rulelist.size() == preds.size() - 1);

    printf("\nOPTIMAL RULE LIST\n");

    // Check for redundant rules 
    int finish = 0;
    int nbRed = 0;
    while(rulelist.size() > 0 && (finish == 0) && (nbRed < rulelist.size())) {
        finish = 1;
        if(strcmp(labels[preds[rulelist.size()-1 - nbRed]].features,
        labels[preds.back()- nbRed].features) == 0) {
            finish = 0;
            nbRed++;
            printf("removing rule : %s => %s because redondant with default decision (%s)\n",
            rules[rulelist[0]].features,
            labels[preds[0]].features,
            labels[preds.back()].features);
        }
    }
    printf("\n");
    if (rulelist.size() > 0 && nbRed != rulelist.size()) {
        printf("if (%s) then (%s)\n", rules[rulelist[0]].features,
               labels[preds[0]].features);
        for (size_t i = 1; i < rulelist.size()-nbRed; ++i) {
            printf("else if (%s) then (%s)\n", rules[rulelist[i]].features,
                   labels[preds[i]].features);
        }
        printf("else (%s)\n\n", labels[preds.back()].features);

        if (latex_out) {
            printf("\nLATEX form of OPTIMAL RULE LIST\n");
            printf("\\begin{algorithmic}\n");
            printf("\\normalsize\n");
            printf("\\State\\bif (%s) \\bthen (%s)\n", rules[rulelist[0]].features,
                   labels[preds[0]].features);
            for (size_t i = 1; i < rulelist.size(); ++i) {
                printf("\\State\\belif (%s) \\bthen (%s)\n", rules[rulelist[i]].features,
                       labels[preds[i]].features);
            }
            printf("\\State\\belse (%s)\n", labels[preds.back()].features);
            printf("\\end{algorithmic}\n\n");
        }
    } else {
        printf("if (1) then (%s)\n\n", labels[preds.back()].features);

        if (latex_out) {
            printf("\nLATEX form of OPTIMAL RULE LIST\n");
            printf("\\begin{algorithmic}\n");
            printf("\\normalsize\n");
            printf("\\State\\bif (1) \\bthen (%s)\n", labels[preds.back()].features);
            printf("\\end{algorithmic}\n\n");
        }
    }

    ofstream f;
    printf("writing optimal rule list to: %s\n\n", fname);
    f.open(fname, ios::out | ios::trunc);
    for(size_t i = 0; i < rulelist.size(); ++i) {
        f << rules[rulelist[i]].features << "~"
          << preds[i] << ";";
    }
    f << "default~" << preds.back();
    f.close();
}

accAndFair computeFinalFairness(int nsamples,
                          const tracking_vector<unsigned short, DataStruct::Tree>& rulelist,
                          const tracking_vector<bool, DataStruct::Tree>& preds,                  
                          rule_t * rules,
                          rule_t * labels,
                          bool isSpecifiedUnpro,
                          int unsensitiveAttrColumn,
                          int sensitiveAttrColumn) {
    accAndFair result;
    // 1) We build the predictions' matrix
    VECTOR captured_it;
    VECTOR not_captured_yet;
    VECTOR captured_zeros;
    VECTOR preds_prefix;
    int nb;
    int nb2;
    int pm;
    rule_vinit(nsamples, &captured_it);
    rule_vinit(nsamples, &not_captured_yet);
    rule_vinit(nsamples, &preds_prefix);
    rule_vinit(nsamples, &captured_zeros);
    // Initially not_captured_yet is full of ones
    rule_vor(not_captured_yet,labels[0].truthtable, labels[1].truthtable, nsamples ,&nb);
    // Initially preds_prefix is full of zeros
    rule_vand(preds_prefix, labels[0].truthtable, labels[1].truthtable, nsamples, &nb);
    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;
    for (size_t i = 0; i < rulelist.size(); ++i) {
        rule_vand(captured_it, not_captured_yet, rules[rulelist[i]].truthtable, nsamples, &nb);
        rule_vandnot(not_captured_yet, not_captured_yet, captured_it, nsamples, &pm);
        rule_vand(captured_zeros, captured_it, labels[0].truthtable, nsamples, &nb2);
        if(nb2 <= (nb - nb2)) { //then prediction is 1
            rule_vor(preds_prefix, preds_prefix, captured_it, nsamples, &nb);
        }
    }
    if(preds.back() == 1) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, not_captured_yet, nsamples, &pm);
    }

    // 2) We compute general metrics for the built predictions matrix

    // true positives, false negatives, true negatives, and false positives tables (for this rule)
    VECTOR A, B, D, C;
    int tp, tn, fp, fn;
    rule_vinit(nsamples, &A);
    rule_vinit(nsamples, &B);
    rule_vinit(nsamples, &D);
    rule_vinit(nsamples, &C);

    rule_vand(A, preds_prefix, labels[1].truthtable, nsamples, &tp);
    rule_vandnot(B, labels[1].truthtable, preds_prefix, nsamples, &fn);
    rule_vandnot(D, labels[0].truthtable, preds_prefix, nsamples, &tn);
    rule_vand(C, preds_prefix, labels[0].truthtable, nsamples, &fp);

    double acc = (double) (tp + tn)/nsamples;
    // 3) We compute group-specific metrics for the model

    VECTOR A_maj, B_maj, D_maj, C_maj;
    rule_vinit(nsamples, &A_maj);
    rule_vinit(nsamples, &B_maj);
    rule_vinit(nsamples, &D_maj);
    rule_vinit(nsamples, &C_maj);

    if(isSpecifiedUnpro) {
        rule_vand(A_maj, A, rules[unsensitiveAttrColumn].truthtable, nsamples, &pm);
        rule_vand(B_maj, B, rules[unsensitiveAttrColumn].truthtable, nsamples, &pm);
        rule_vand(D_maj, D, rules[unsensitiveAttrColumn].truthtable, nsamples, &pm);
        rule_vand(C_maj, C, rules[unsensitiveAttrColumn].truthtable, nsamples, &pm);
    } else {
        rule_vandnot(A_maj, A, rules[sensitiveAttrColumn].truthtable, nsamples, &pm);
        rule_vandnot(B_maj, B, rules[sensitiveAttrColumn].truthtable, nsamples, &pm);
        rule_vandnot(D_maj, D, rules[sensitiveAttrColumn].truthtable, nsamples, &pm);
        rule_vandnot(C_maj, C, rules[sensitiveAttrColumn].truthtable, nsamples, &pm);
    }

    VECTOR A_min, B_min, D_min, C_min;
    rule_vinit(nsamples, &A_min);
    rule_vinit(nsamples, &B_min);
    rule_vinit(nsamples, &D_min);
    rule_vinit(nsamples, &C_min);
    
    rule_vand(A_min, A, rules[sensitiveAttrColumn].truthtable, nsamples, &pm);
    rule_vand(B_min, B, rules[sensitiveAttrColumn].truthtable, nsamples, &pm);
    rule_vand(D_min, D, rules[sensitiveAttrColumn].truthtable, nsamples, &pm);
    rule_vand(C_min, C, rules[sensitiveAttrColumn].truthtable, nsamples, &pm);

    int nA_maj = mpz_popcount(A_maj);
    int nB_maj = mpz_popcount(B_maj);
    int nD_maj = mpz_popcount(D_maj);
    int nC_maj = mpz_popcount(C_maj);

    int nA_min = mpz_popcount(A_min);
    int nB_min = mpz_popcount(B_min);
    int nD_min = mpz_popcount(D_min);
    int nC_min = mpz_popcount(C_min);

    //printf("maj : tp = %d, tn = %d, fp = %d, fn = %d\n", nA_maj, nD_maj, nC_maj, nB_maj);
    //printf("min : tp = %d, tn = %d, fp = %d, fn = %d\n", nA_min, nD_min, nC_min, nB_min);
    
    double statistical_parity_maj_0 = (double) (nA_maj + nC_maj)/max((nA_maj + nB_maj + nC_maj + nD_maj),1);
    double statistical_parity_maj_1 = (double) (nB_maj + nD_maj)/max((nA_maj + nB_maj + nC_maj + nD_maj),1);
    double statistical_parity_min_0 = (double) (nA_min + nC_min)/max((nA_min + nB_min + nC_min + nD_min),1);
    double statistical_parity_min_1 = (double) (nB_min + nD_min)/max((nA_min + nB_min + nC_min + nD_min),1);
    double stat_par_sum = fabs(statistical_parity_maj_0 - statistical_parity_min_0);// + fabs(statistical_parity_maj_1 - statistical_parity_min_1);
    
    // Free allocated VECTORS
    rule_vfree(&captured_it);
    rule_vfree(&not_captured_yet);
    rule_vfree(&captured_zeros);
    rule_vfree(&preds_prefix);
    rule_vfree(&A);
    rule_vfree(&B);
    rule_vfree(&D);
    rule_vfree(&C);
    rule_vfree(&A_maj);
    rule_vfree(&B_maj);
    rule_vfree(&D_maj);
    rule_vfree(&C_maj);
    rule_vfree(&A_min);
    rule_vfree(&B_min);
    rule_vfree(&D_min);
    rule_vfree(&C_min);

    result.accuracy = acc;
    result.fairness = 1-stat_par_sum;
    return result;
}

/*
 * Prints out information about the machine.
 */
void print_machine_info() {
    struct utsname buffer;

    if (uname(&buffer) == 0) {
        printf("System information:\n"
               "system name-> %s; node name-> %s; release-> %s; "
               "version-> %s; machine-> %s\n\n",
               buffer.sysname,
               buffer.nodename,
               buffer.release,
               buffer.version,
               buffer.machine);
    }
}
