#include "queue.hh"
#include <algorithm>
#include <iostream>
#include <sys/resource.h>
#include <stdio.h>
#include <math.h>
#include <time.h> // for execution time measurements

extern int sensitiveAttrColumn;
extern float beta;
extern double min_fairness_acceptable;
extern int mode;
extern int unsensitiveAttrColumn;
extern bool isSpecifiedUnpro;
extern int fairnessBoundPruneCnt;
extern long totalFairTime;
extern bool useUnfairnessLB;
int first = 1;
int pushingTicket = 0;
double squareCalc(double d1) {
    return d1*d1;
}

fairness_metrics compute_fairness_metrics(VECTOR preds_prefix_parent, tracking_vector<unsigned short, DataStruct::Tree> parent_prefix, CacheTree* tree, VECTOR parent_not_captured, VECTOR captured, int pred, int index, int defaultPred){
    long int timemillis0 = clock()/CLOCKS_PER_SEC;
    fairness_metrics all_metrics;
//  printf("parents not captured : %d, captured : %d\n", parent_not_captured->_mp_size, captured->_mp_size);
//  printf("nsamples : %d\n", tree->nsamples());
//  printf("nrules = %d\n", tree->nrules());
    VECTOR not_captured;
    int num_not_captured;
    rule_vinit(tree->nsamples(), &not_captured);
    rule_vandnot(not_captured, parent_not_captured, captured, tree->nsamples(), &num_not_captured);
    //VECTOR not_captured_yet;
    VECTOR preds_prefix;
    //rule_vinit(tree->nsamples(), &not_captured_yet);
    rule_vinit(tree->nsamples(), &preds_prefix);
    //printf("preds prefix parent = %d, pred = %d, default pred = %d\n",mpz_popcount(preds_prefix_parent), pred, defaultPred);
    /*
    VECTOR A_rule, B_rule, D_rule, C_rule;
    rule_vinit(tree->nsamples(), &A_rule);
    rule_vinit(tree->nsamples(), &B_rule);
    rule_vinit(tree->nsamples(), &D_rule);
    rule_vinit(tree->nsamples(), &C_rule);

    VECTOR A_default, B_default, D_default, C_default;
    rule_vinit(tree->nsamples(), &A_default);
    rule_vinit(tree->nsamples(), &B_default);
    rule_vinit(tree->nsamples(), &D_default);
    rule_vinit(tree->nsamples(), &C_default);
    */
    int pm;
    //rule_vandnot(not_captured_yet, not_captured_yet_parent, captured, tree->nsamples(), &pm);

    rule_copy(preds_prefix, preds_prefix_parent,  tree->nsamples());
    if(defaultPred == 1) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, not_captured, tree->nsamples(), &pm);
    }
    //printf("preds prefix = %d\n",mpz_popcount(preds_prefix));

    if(pred == 1) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, captured, tree->nsamples(), &pm);
    }

    //printf("preds prefix = %d\n",mpz_popcount(preds_prefix));


    // true positives, false negatives, true negatives, and false positives tables (for this rule)
    VECTOR A, B, D, C;
    rule_vinit(tree->nsamples(), &A);
    rule_vinit(tree->nsamples(), &B);
    rule_vinit(tree->nsamples(), &D);
    rule_vinit(tree->nsamples(), &C);

    rule_vand(A, preds_prefix, tree->label(1).truthtable, tree->nsamples(), &pm);
    //printf("TP = %d ", pm);
    rule_vandnot(B, tree->label(1).truthtable, preds_prefix, tree->nsamples(), &pm);
    //printf("FN = %d ", pm);
    rule_vandnot(D, tree->label(0).truthtable, preds_prefix, tree->nsamples(), &pm);
    //printf("TN = %d ", pm);
    rule_vand(C, preds_prefix, tree->label(0).truthtable, tree->nsamples(), &pm);
    //printf("FP = %d\n", pm);

    // true positives, false negatives, true negatives, and false positives tables for majority group
    VECTOR A_maj, B_maj, D_maj, C_maj;
    rule_vinit(tree->nsamples(), &A_maj);
    rule_vinit(tree->nsamples(), &B_maj);
    rule_vinit(tree->nsamples(), &D_maj);
    rule_vinit(tree->nsamples(), &C_maj);

    if(isSpecifiedUnpro) {
        mpz_and(A_maj, A, tree->rule(unsensitiveAttrColumn).truthtable);
        mpz_and(B_maj, B, tree->rule(unsensitiveAttrColumn).truthtable);
        mpz_and(D_maj, D, tree->rule(unsensitiveAttrColumn).truthtable);
        mpz_and(C_maj, C, tree->rule(unsensitiveAttrColumn).truthtable);
    } else {
        rule_vandnot(A_maj, A, tree->rule(sensitiveAttrColumn).truthtable, tree->nsamples(), &pm);
        rule_vandnot(B_maj, B, tree->rule(sensitiveAttrColumn).truthtable, tree->nsamples(), &pm);
        rule_vandnot(D_maj, D, tree->rule(sensitiveAttrColumn).truthtable, tree->nsamples(), &pm);
        rule_vandnot(C_maj, C, tree->rule(sensitiveAttrColumn).truthtable, tree->nsamples(), &pm);
    }
    if(first) {
        first = 0;
        printf("sensitive attr is : %s\n", tree->rule(sensitiveAttrColumn).features);
        if(isSpecifiedUnpro) {
            printf("unsensitive attr is : %s\n", tree->rule(unsensitiveAttrColumn).features);
        } else {
            printf("unsensitive attr is : not(%s)\n", tree->rule(sensitiveAttrColumn).features);
        }
    }
    // true positives, false negatives, true negatives, and false positives tables for minority group
    VECTOR A_min, B_min, D_min, C_min;
    rule_vinit(tree->nsamples(), &A_min);
    rule_vinit(tree->nsamples(), &B_min);
    rule_vinit(tree->nsamples(), &D_min);
    rule_vinit(tree->nsamples(), &C_min);
    
    mpz_and(A_min, A, tree->rule(sensitiveAttrColumn).truthtable);
    mpz_and(B_min, B, tree->rule(sensitiveAttrColumn).truthtable);
    mpz_and(D_min, D, tree->rule(sensitiveAttrColumn).truthtable);
    mpz_and(C_min, C, tree->rule(sensitiveAttrColumn).truthtable);

    int nA_maj = mpz_popcount(A_maj);
    int nB_maj = mpz_popcount(B_maj);
    int nD_maj = mpz_popcount(D_maj);
    int nC_maj = mpz_popcount(C_maj);

    int nA_min = mpz_popcount(A_min);
    int nB_min = mpz_popcount(B_min);
    int nD_min = mpz_popcount(D_min);
    int nC_min = mpz_popcount(C_min);

    // overall accuracy equality
    /*double overall_accuracy_equality_maj = (double) (nA_maj + nD_maj)/(nA_maj + nB_maj + nC_maj + nD_maj);
    double overall_accuracy_equality_min = (double) (nA_min + nD_min)/(nA_min + nB_min + nC_min + nD_min);
    all_metrics.overall_accuracy_equality = fabs(overall_accuracy_equality_maj - overall_accuracy_equality_min);
    */
    // statistical parity
    double statistical_parity_maj_0 = (double) (nA_maj + nC_maj)/max((nA_maj + nB_maj + nC_maj + nD_maj),1);
    double statistical_parity_maj_1 = (double) (nB_maj + nD_maj)/max((nA_maj + nB_maj + nC_maj + nD_maj),1);
    double statistical_parity_min_0 = (double) (nA_min + nC_min)/max((nA_min + nB_min + nC_min + nD_min),1);
    double statistical_parity_min_1 = (double) (nB_min + nD_min)/max((nA_min + nB_min + nC_min + nD_min),1);
    all_metrics.statistical_parity_sum = fabs(statistical_parity_maj_0 - statistical_parity_min_0);// + fabs(statistical_parity_maj_1 - statistical_parity_min_1);
    all_metrics.statistical_parity_max = max(fabs(statistical_parity_maj_0 - statistical_parity_min_0), fabs(statistical_parity_maj_1 - statistical_parity_min_1));
   
   
    int nA_maj_ub;
    int nB_maj_ub;
    int nD_maj_ub;
    int nC_maj_ub;

   /* UPPER BOUND CALC */
    VECTOR A_min_ub, B_min_ub, D_min_ub, C_min_ub;
    rule_vinit(tree->nsamples(), &A_min_ub);
    rule_vinit(tree->nsamples(), &B_min_ub);
    rule_vinit(tree->nsamples(), &D_min_ub);
    rule_vinit(tree->nsamples(), &C_min_ub);
    VECTOR A_maj_ub, B_maj_ub, D_maj_ub, C_maj_ub;
    rule_vinit(tree->nsamples(), &A_maj_ub);
    rule_vinit(tree->nsamples(), &B_maj_ub);
    rule_vinit(tree->nsamples(), &D_maj_ub);
    rule_vinit(tree->nsamples(), &C_maj_ub);

    rule_vandnot(A_maj_ub, A_maj, not_captured, tree->nsamples(), &nA_maj_ub);
    rule_vandnot(B_maj_ub, B_maj, not_captured, tree->nsamples(), &nB_maj_ub);
    rule_vandnot(C_maj_ub, C_maj, not_captured, tree->nsamples(), &nC_maj_ub);
    rule_vandnot(D_maj_ub, D_maj, not_captured, tree->nsamples(), &nD_maj_ub);

    int nA_min_ub;
    int nB_min_ub;
    int nD_min_ub;
    int nC_min_ub;

    rule_vandnot(A_min_ub, A_min, not_captured, tree->nsamples(), &nA_min_ub);
    rule_vandnot(B_min_ub, B_min, not_captured, tree->nsamples(), &nB_min_ub);
    rule_vandnot(C_min_ub, C_min, not_captured, tree->nsamples(), &nC_min_ub);
    rule_vandnot(D_min_ub, D_min, not_captured, tree->nsamples(), &nD_min_ub);

    //if(nA_maj_ub)
/*
    int nA_maj_ub0 = mpz_popcount(A_maj_ub);
    int nB_maj_ub0 = mpz_popcount(B_maj_ub);
    int nD_maj_ub0 = mpz_popcount(D_maj_ub);
    int nC_maj_ub0 = mpz_popcount(C_maj_ub);

    int nA_min_ub0 = mpz_popcount(A_min_ub);
    int nB_min_ub0 = mpz_popcount(B_min_ub);
    int nD_min_ub0 = mpz_popcount(D_min_ub);
    int nC_min_ub0 = mpz_popcount(C_min_ub);

    if( nA_maj_ub != nA_maj_ub0 || nB_maj_ub != nB_maj_ub0 || nC_maj_ub != nC_maj_ub0 || nD_maj_ub != nD_maj_ub0 || nA_min_ub != nA_min_ub0 || nB_min_ub != nB_min_ub0 || nC_min_ub != nC_min_ub0 || nD_min_ub != nD_min_ub0) {
        printf("PROBLEM");
        exit(-1);
    }
*/
    // overall accuracy equality
    /*double overall_accuracy_equality_maj = (double) (nA_maj + nD_maj)/(nA_maj + nB_maj + nC_maj + nD_maj);
    double overall_accuracy_equality_min = (double) (nA_min + nD_min)/(nA_min + nB_min + nC_min + nD_min);
    all_metrics.overall_accuracy_equality = fabs(overall_accuracy_equality_maj - overall_accuracy_equality_min);
    */

    // statistical parity
    //double statistical_parity_maj_0_ub = (double) (nA_maj_ub + nC_maj_ub)/max((nA_maj_ub + nB_maj_ub + nC_maj_ub + nD_maj_ub),1);
    //double statistical_parity_maj_1_ub = (double) (nB_maj_ub + nD_maj_ub)/max((nA_maj_ub + nB_maj_ub + nC_maj_ub + nD_maj_ub),1);
    //double statistical_parity_min_0_ub = (double) (nA_min_ub + nC_min_ub)/max((nA_min_ub + nB_min_ub + nC_min_ub + nD_min_ub),1);
    //double statistical_parity_min_1_ub = (double) (nB_min_ub + nD_min_ub)/max((nA_min_ub + nB_min_ub + nC_min_ub + nD_min_ub),1);
    //double statistical_parity_sum_ub = fabs(statistical_parity_maj_0_ub - statistical_parity_min_0_ub);// + fabs(statistical_parity_maj_1_ub - statistical_parity_min_1_ub);
    
    int totMAJ = nA_maj + nB_maj + nC_maj + nD_maj;
    int totMIN = nA_min + nB_min + nC_min + nD_min;
  /*  if((totMAJ+totMIN-(nA_maj_ub + nB_maj_ub + nC_maj_ub + nD_maj_ub + nA_min_ub + nB_min_ub + nC_min_ub + nD_min_ub))!= num_not_captured) {
        printf("problem\n");
        exit(-1);
    }*/
    double B1 = (double)((double)(totMAJ - nB_maj_ub - nD_maj_ub)/(double)totMAJ);
    double B2 = (double) ((double)(nA_maj_ub + nC_maj_ub)/(double)totMAJ);
    double B3 = (double) ((double)(totMIN - nB_min_ub - nD_min_ub)/(double)totMIN);
    double B4 = (double) ((double)(nA_min_ub + nC_min_ub)/(double)totMIN);
    double min_val = 0;
    if(B1 < B2 || B3 < B4) {
        printf("problem !\n");
        exit(-1);
    }
    if(B3 < B2) {
        min_val = (B2-B3);
    } else if(B4 > B1) {
        min_val = (B4-B1);
    } else {
        min_val = 0;
    }
   /* if(min_val>0) {
        printf("B2 = %lf, B1 = %lf, B4 = %lf, B3 = %lf, lower bound = %lf\n", B2,B1,B4,B3,min_val);
   }*/
    all_metrics.lowerbound_unfairness = min_val;
   // int totConcern = nA_maj + nB_maj + nC_maj + nD_maj + nA_min + nB_min + nC_min + nD_min;
   // printf("TOTAL COUNT : %d / %d \n",(nA_maj_ub + nB_maj_ub + nC_maj_ub + nD_maj_ub + nA_min_ub + nB_min_ub + nC_min_ub + nD_min_ub),totConcern);

   // printf("Total rate of captured instances : %lf, partial unfairness = %lf\n", (double) (nA_maj_ub + nB_maj_ub + nC_maj_ub + nD_maj_ub + nA_min_ub + nB_min_ub + nC_min_ub + nD_min_ub)/tree->nsamples(),statistical_parity_sum_ub);
   // all_metrics.lowerbound_unfairness = (double) ((double)(nA_maj_ub + nB_maj_ub + nC_maj_ub + nD_maj_ub + nA_min_ub + nB_min_ub + nC_min_ub + nD_min_ub)/totConcern)*statistical_parity_sum_ub;
   // rule_vfree(&captured_yet);
    //rule_vfree(&testTool2);
  //  printf("unfairness lower bound = %lf\n", all_metrics.lowerbound_unfairness);
   //if(isnan(all_metrics.statistical_parity_sum) || (all_metrics.statistical_parity_sum == 0) && (pred == 1)) {
    //printf("rule = %s, TP_maj = %d, FP_maj = %d, TP_min = %d, FP_min = %d, FN_maj = %d, TN_maj = %d, FN_min = %d, TN_min = %d, total_maj = %d, total_min = %d\n", tree->rule(index).features, nA_maj, nC_maj, nA_min, nC_min ,nB_maj, nD_maj, nB_min, nD_min, (nA_maj + nB_maj + nC_maj + nD_maj), (nA_min + nB_min + nC_min + nD_min));
   //}
   
   //if(((nA_min + nB_min + nC_min + nD_min) != 4883) || ((nA_maj + nB_maj + nC_maj + nD_maj) != 10041))
   //     printf("TP_maj = %d, FP_maj = %d, TP_min = %d, FP_min = %d, total_maj = %d, total_min = %d\n", nA_maj, nC_maj, nA_min, nC_min, (nA_maj + nB_maj + nC_maj + nD_maj), (nA_min + nB_min + nC_min + nD_min));
/* 
    // conditional procedure accuracy equality
    double conditional_procedure_accuracy_equality_maj_0 = (double) (nA_maj)/(nA_maj + nB_maj);
    double conditional_procedure_accuracy_equality_maj_1 = (double) (nD_maj)/(nC_maj + nD_maj);
    double conditional_procedure_accuracy_equality_min_0 = (double) (nA_min)/(nA_min + nB_min);
    double conditional_procedure_accuracy_equality_min_1 = (double) (nD_min)/(nC_min + nD_min);
    all_metrics.conditional_procedure_accuracy_equality_sum = fabs(conditional_procedure_accuracy_equality_maj_0 - conditional_procedure_accuracy_equality_min_0) + fabs(conditional_procedure_accuracy_equality_maj_1 - conditional_procedure_accuracy_equality_min_1);
    all_metrics.conditional_procedure_accuracy_equality_max = max(fabs(conditional_procedure_accuracy_equality_maj_0 - conditional_procedure_accuracy_equality_min_0), fabs(conditional_procedure_accuracy_equality_maj_1 - conditional_procedure_accuracy_equality_min_1));


    // conditional use accuracy equality
    double conditional_use_accuracy_equality_maj_0 = (double) (nA_maj)/(nA_maj + nC_maj);
    double conditional_use_accuracy_equality_maj_1 = (double) (nD_maj)/(nB_maj + nD_maj);
    double conditional_use_accuracy_equality_min_0 = (double) (nA_min)/(nA_min + nC_min);
    double conditional_use_accuracy_equality_min_1 = (double) (nD_min)/(nB_min + nD_min);
    all_metrics.conditional_use_accuracy_equality_sum = fabs(conditional_use_accuracy_equality_maj_0 - conditional_use_accuracy_equality_min_0) + fabs(conditional_use_accuracy_equality_maj_1 - conditional_use_accuracy_equality_min_1);
    all_metrics.conditional_use_accuracy_equality_max = max(fabs(conditional_use_accuracy_equality_maj_0 - conditional_use_accuracy_equality_min_0), fabs(conditional_use_accuracy_equality_maj_1 - conditional_use_accuracy_equality_min_1));


    // treatment equality
    double treatment_maj;
    if (nB_maj != 0){
        treatment_maj = (double) nC_maj / nB_maj;
    } else {
        treatment_maj = 0.0;
    }

    double treatment_min;
    if (nB_min != 0){
        treatment_min = (double) nC_min / nB_min;
    } else {
        treatment_min = 0.0;
    }

    all_metrics.treatment_equality = fabs(treatment_maj - treatment_min);
    //all_metrics.treatment_equality = max(treatment_maj, treatment_min);
*/
    rule_vfree(&not_captured);
    rule_vfree(&A);
    rule_vfree(&B);
    rule_vfree(&D);
    rule_vfree(&C);
    rule_vfree(&preds_prefix);
  /*rule_vfree(&A_rule);
    rule_vfree(&B_rule);
    rule_vfree(&D_rule);
    rule_vfree(&C_rule);
    
    rule_vfree(&A_default);
    rule_vfree(&B_default);
    rule_vfree(&D_default);
    rule_vfree(&C_default);*/

    rule_vfree(&A_maj);
    rule_vfree(&B_maj);
    rule_vfree(&D_maj);
    rule_vfree(&C_maj);
    rule_vfree(&A_min);
    rule_vfree(&B_min);
    rule_vfree(&D_min);
    rule_vfree(&C_min);

    rule_vfree(&A_maj_ub);
    rule_vfree(&B_maj_ub);
    rule_vfree(&D_maj_ub);
    rule_vfree(&C_maj_ub);
    
    rule_vfree(&A_min_ub);
    rule_vfree(&B_min_ub);
    rule_vfree(&D_min_ub);
    rule_vfree(&C_min_ub);
    //return (statistical_parity < 0.01) ? 0 : statistical_parity;
    long int timemillis1 = clock()/CLOCKS_PER_SEC;
    totalFairTime+=(timemillis1-timemillis0);
    return all_metrics;

}

Queue::Queue(std::function<bool(Node*, Node*)> cmp, char const *type)
    : q_(new q (cmp)), type_(type) {}

/*
 * Performs incremental computation on a node, evaluating the bounds and inserting into the cache,
 * queue, and permutation map if appropriate.
 * This is the function that contains the majority of the logic of the algorithm.
 *
 * parent -- the node that is going to have all of its children evaluated.
 * parent_not_captured -- the vector representing data points NOT captured by the parent.
 */
void evaluate_children(CacheTree* tree, Node* parent, tracking_vector<unsigned short, DataStruct::Tree> parent_prefix,
    VECTOR parent_not_captured, Queue* q, PermutationMap* p) {
    VECTOR captured, captured_zeros, not_captured, not_captured_zeros, not_captured_equivalent;
    int num_captured, c0, c1, captured_correct;
    int num_not_captured, d0, d1, default_correct, num_not_captured_equivalent;
    bool prediction, default_prediction;
    double lower_bound, objective, parent_lower_bound, lookahead_bound;
    double parent_equivalent_minority;
    double equivalent_minority = 0.;
    int nsamples = tree->nsamples();
    int nrules = tree->nrules();
    double c = tree->c();
    double threshold = c * nsamples;
    rule_vinit(nsamples, &captured);
    rule_vinit(nsamples, &captured_zeros);
    rule_vinit(nsamples, &not_captured);
    rule_vinit(nsamples, &not_captured_zeros);
    rule_vinit(nsamples, &not_captured_equivalent);
    int i, len_prefix;
    len_prefix = parent->depth() + 1;
    parent_lower_bound = parent->lower_bound();
    parent_equivalent_minority = parent->equivalent_minority();
    double t0 = timestamp();

    // Compute prefix's predictions
    VECTOR captured_it;
    VECTOR not_captured_yet;
    VECTOR captured_zeros_j;
    VECTOR preds_prefix;
    int nb;
    int nb2;
    int pm;
    rule_vinit(tree->nsamples(), &captured_it);
    rule_vinit(tree->nsamples(), &not_captured_yet);
    rule_vinit(tree->nsamples(), &preds_prefix);
    rule_vinit(tree->nsamples(), &captured_zeros_j);
    // Initially not_captured_yet is full of ones
    rule_vor(not_captured_yet, tree->label(0).truthtable, tree->label(1).truthtable, tree->nsamples(),&nb);
    // Initially preds_prefix is full of zeros
    //rule_vand(preds_prefix, tree->label(0).truthtable, tree->label(1).truthtable, tree->nsamples(),&nb);
    mpz_set_ui(preds_prefix, 0);
    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;
    //printf("--------------------------------------------------------------\n");
    for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
        //printf("precedent rules : %s\n", tree->rule(*it).features);
        rule_vand(captured_it, not_captured_yet, tree->rule(*it).truthtable, tree->nsamples(), &nb);
        rule_vandnot(not_captured_yet, not_captured_yet, captured_it, tree->nsamples(), &pm);
        rule_vand(captured_zeros_j, captured_it, tree->label(0).truthtable, tree->nsamples(), &nb2);
        if(nb2 <= (nb - nb2)) { //then prediction is 1
            rule_vor(preds_prefix, preds_prefix, captured_it, tree->nsamples(), &nb);
        }
    }
    for (i = 1; i < nrules; i++) { // Loop on children nodes of current rule
        double t1 = timestamp();
        // check if this rule is already in the prefix
        if (std::find(parent_prefix.begin(), parent_prefix.end(), i) != parent_prefix.end())
            continue;
        // captured represents data captured by the new rule
        rule_vand(captured, parent_not_captured, tree->rule(i).truthtable, nsamples, &num_captured);
        // lower bound on antecedent support
        if ((tree->ablation() != 1 && tree->ablation() != 3) && (num_captured < threshold))
            continue;
        rule_vand(captured_zeros, captured, tree->label(0).truthtable, nsamples, &c0);
        c1 = num_captured - c0;
        if (c0 > c1) {
            prediction = 0;
            captured_correct = c0;
        } else {
            prediction = 1;
            captured_correct = c1;
        }
        //int rule_misc = num_captured - captured_correct;
        // lower bound on accurate antecedent support
        if ((tree->ablation() != 1 && tree->ablation() != 3) && (captured_correct < threshold))
            continue;
        // subtract off parent equivalent points bound because we want to use pure lower bound from parent
        lower_bound = parent_lower_bound - parent_equivalent_minority + (double)(num_captured - captured_correct) / nsamples + c;
        logger->addToLowerBoundTime(time_diff(t1));
        logger->incLowerBoundNum();
        if (lower_bound >= tree->min_objective()) // hierarchical objective lower bound
	        continue;
        double t2 = timestamp();
        rule_vandnot(not_captured, parent_not_captured, captured, nsamples, &num_not_captured);
        rule_vand(not_captured_zeros, not_captured, tree->label(0).truthtable, nsamples, &d0);
        d1 = num_not_captured - d0;
        if (d0 > d1) {
            default_prediction = 0;
            default_correct = d0;
        } else {
            default_prediction = 1;
            default_correct = d1;
        }
        /* Objective function evaluation, depending on the given mode */
      //  printf("num_captured = %lf\n", (double)(num_captured));
     //   printf("num_not_captured = %d, default_correct = %d\n", num_not_captured, default_correct);
        if(mode == 2) { // Max fairness
            double unfairness = compute_fairness_metrics(preds_prefix, parent_prefix, tree, parent_not_captured, captured, prediction, i, default_prediction).statistical_parity_sum;
            objective = unfairness + lower_bound;
        } else if(mode == 3) {
            //double unfairness = compute_fairness_metrics(parent_prefix, tree, parent_not_captured, captured, prediction, i, default_prediction).statistical_parity_sum;
          /*  if(1 - unfairness < min_fairness_acceptable) {
                //printf("Overunfairness!\n");
                objective = 10000000;
            } else {*/
                //printf("%lf\n", 1-unfairness);
                double misc = (double)(num_not_captured - default_correct) / nsamples;
                objective =  misc + lower_bound;
            //}
        } else if(mode == 4) {
                double misc = (double)(num_not_captured - default_correct) / nsamples;
                objective =  misc + lower_bound;
        } else {
            // Modification que j'ai faite : rule_misc est le nombre d'instances 
            // mal classifiées par la règle en cours d'évaluation
            double misc = (double)(num_not_captured - default_correct) / nsamples;
            //tree->accTemp = (double) ((double) 1 - (double)((double)(num_not_captured-default_correct)/(double)nsamples));
            //printf("misc = %lf", misc);
            double unfairness = compute_fairness_metrics(preds_prefix, parent_prefix, tree, parent_not_captured, captured, prediction, i, default_prediction).statistical_parity_sum;
            if(first) {
                //printf("beta = %lf\n", beta);
                printf("1-beta = %lf\n", 1 - beta);
                printf("misc = %lf\n", misc);
                printf("1-beta*misc = %lf\n", (1-beta)*misc);
                printf("beta*unfairness = %lf\n",beta*unfairness);
                first = 0;
            }
            /* Distance-to-a-reference objective function */
            /*double unfairnessObjective = 0.0;
            double accuracyObjective = 1.0;
            double distance = sqrt((beta*squareCalc(unfairness - unfairnessObjective)) + ((1-beta)*squareCalc((1 - misc) - accuracyObjective)));
            objective = distance + lower_bound;*/

            /* Weighted sum of objective functions */
            //objective = distance + lower_bound;
            objective =  (1-beta)*misc + beta*unfairness + lower_bound;
    
            //objective =  misc + lower_bound;
            /* Original objective function */
            //objective = lower_bound + (double)(num_not_captured - default_correct) / nsamples;
        }
        
        double fairnesslb = 1.0;
        double unfairness = 0.0;
        if(mode == 3) {
            fairness_metrics fm = compute_fairness_metrics(preds_prefix, parent_prefix, tree, not_captured, captured, prediction, i, default_prediction);
            unfairness = fm.statistical_parity_sum;
            if(useUnfairnessLB) {
                fairnesslb = 1 - (fm.lowerbound_unfairness);   
                if(fairnesslb<min_fairness_acceptable) {
                 /*   printf("PRUNING PREFIX -------------------------------------------\n");
                    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;
                    for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
                        printf("%s\n", tree->rule(*it).features);
                    }
                    printf("%s\n",tree->rule(i).features); 
                    printf("fairnesslb = %lf\n", fairnesslb);*/
                    fairnessBoundPruneCnt++;
                }
            }
        }
        logger->addToObjTime(time_diff(t2));
        logger->incObjNum();
        if (objective < tree->min_objective()) {
             if(mode == 3) {
                //printf("unfairness = %lf\n", unfairness);
                //printf("(1-unfairness) = %lf, min_fairness_acceptable = %lf\n",(1-unfairness),min_fairness_acceptable);
                if((1-unfairness) > min_fairness_acceptable) {
                    printf("min(objective): %1.5f -> %1.5f, length: %d, cache size: %zu\n",
                    tree->min_objective(), objective, len_prefix, tree->num_nodes());
                    printf("(1-unfairness) = %lf, min_fairness_acceptable = %lf\n",(1-unfairness),min_fairness_acceptable);
                    logger->setTreeMinObj(objective);
                    tree->update_min_objective(objective);
                    tree->update_opt_rulelist(parent_prefix, i);
                    tree->update_opt_predictions(parent, prediction, default_prediction);
                    // dump state when min objective is updated
                    logger->dumpState();
                    //printf("fairness value = %lf\n", 1-unfairness);
                /*  printf("new best solution :\n");
                    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;
                    for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
                        printf("%s\n", tree->rule(*it).features);
                    }
                    printf("%s\n",tree->rule(i).features); */
                }
            } else {
            //  double unfairness = compute_fairness_metrics(parent_prefix, tree, not_captured, captured, prediction, i, default_prediction).statistical_parity_sum;
                
                printf("min(objective): %1.5f -> %1.5f, length: %d, cache size: %zu\n",
                tree->min_objective(), objective, len_prefix, tree->num_nodes());

            // printf("(1-unfairness) = %lf, min_fairness_acceptable = %lf\n",(1-unfairness),min_fairness_acceptable);

                logger->setTreeMinObj(objective);
                tree->update_min_objective(objective);
                tree->update_opt_rulelist(parent_prefix, i);
                tree->update_opt_predictions(parent, prediction, default_prediction);
                // dump state when min objective is updated
                logger->dumpState();
            }
        }
        // calculate equivalent points bound to capture the fact that the minority points can never be captured correctly
        if (tree->has_minority()) {
            rule_vand(not_captured_equivalent, not_captured, tree->minority(0).truthtable, nsamples, &num_not_captured_equivalent);
            equivalent_minority = (double)(num_not_captured_equivalent) / nsamples;
            lower_bound += equivalent_minority;
        }
        if (tree->ablation() != 2 && tree->ablation() != 3)
            lookahead_bound = lower_bound + c;
        else
            lookahead_bound = lower_bound;
        // only add node to our datastructures if its children will be viable
      /*   if(fairnesslb<min_fairness_acceptable) {
            printf("Fairness lower bound = %lf, min acceptable = %lf => pruning %s\n", fairnesslb, min_fairness_acceptable, tree->rule(i).features);
      }*/
        if ((lookahead_bound < tree->min_objective()) && (fairnesslb>=min_fairness_acceptable)) {
            double t3 = timestamp();
            // check permutation bound
            Node* n = p->insert(i, nrules, prediction, default_prediction,
                                   lower_bound, objective, parent, num_not_captured, nsamples,
                                   len_prefix, c, equivalent_minority, tree, not_captured, parent_prefix);
            logger->addToPermMapInsertionTime(time_diff(t3));
            // n is NULL if this rule fails the permutaiton bound
            if (n) {
                pushingTicket++;
                n->set_num(pushingTicket);
                double t4 = timestamp();
                tree->insert(n);
                logger->incTreeInsertionNum();
                logger->incPrefixLen(len_prefix);
                logger->addToTreeInsertionTime(time_diff(t4));
                double t5 = timestamp();
                q->push(n);
                logger->setQueueSize(q->size());
                if (tree->calculate_size())
                    logger->addQueueElement(len_prefix, lower_bound, false);
                logger->addToQueueInsertionTime(time_diff(t5));
            }
        } // else:  objective lower bound with one-step lookahead 
      /*  else if(fairnesslb < min_fairness_acceptable) {
            if(0 == strcmp(tree->rule(*(parent_prefix.begin())).features, "{capital_gain:<57.0}")) {
                printf("unsufficient bound :\n");
                tracking_vector<unsigned short, DataStruct::Tree>::iterator it;
                for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
                    printf("%s\n", tree->rule(*it).features);
                }
                printf("%s\n",tree->rule(i).features); 
                printf("fairnesslb = %lf\n", fairnesslb);
            }   
        }*/
    }
    rule_vfree(&captured_it);
    rule_vfree(&not_captured_yet);
    rule_vfree(&preds_prefix);

    rule_vfree(&captured);
    rule_vfree(&captured_zeros);
    rule_vfree(&captured_zeros_j);
    rule_vfree(&not_captured);
    rule_vfree(&not_captured_zeros);
    rule_vfree(&not_captured_equivalent);

    logger->addToRuleEvalTime(time_diff(t0));
    logger->incRuleEvalNum();
    logger->decPrefixLen(parent->depth());
    if (tree->calculate_size())
        logger->removeQueueElement(len_prefix - 1, parent_lower_bound, false);
    if (parent->num_children() == 0) {
      /*/  if(0 == strcmp(tree->rule(*(parent_prefix.begin())).features, "{capital_gain:<57.0}")) {
            printf("removing parent node from prefix :\n");
            tracking_vector<unsigned short, DataStruct::Tree>::iterator it;
            for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
                printf("%s\n", tree->rule(*it).features);
            }
        }*/
        tree->prune_up(parent);
    } else {
        parent->set_done();
        tree->increment_num_evaluated();
    }
}

/*
 * Explores the search space by using a queue to order the search process.
 * The queue can be ordered by DFS, BFS, or an alternative priority metric (e.g. lower bound).
 */
int bbound(CacheTree* tree, size_t max_num_nodes, Queue* q, PermutationMap* p) {
    bool print_queue = 0;
    size_t num_iter = 0;
    int cnt;
    double min_objective;
    VECTOR captured, not_captured;
    rule_vinit(tree->nsamples(), &captured);
    rule_vinit(tree->nsamples(), &not_captured);

    size_t queue_min_length = logger->getQueueMinLen();

    double start = timestamp();
    logger->setInitialTime(start);
    logger->initializeState(tree->calculate_size());
    int verbosity = logger->getVerbosity();
    // initial log record
    logger->dumpState();         

    min_objective = 1.0;
    tree->insert_root();
    logger->incTreeInsertionNum();
    tree->root()->set_num(0);
    q->push(tree->root());
    logger->setQueueSize(q->size());
    logger->incPrefixLen(0);
    // log record for empty rule list
    logger->dumpState();
    while ((tree->num_nodes() < max_num_nodes) && !q->empty()) {
        double t0 = timestamp();
        std::pair<Node*, tracking_vector<unsigned short, DataStruct::Tree> > node_ordered = q->select(tree, captured);
        logger->addToNodeSelectTime(time_diff(t0));
        logger->incNodeSelectNum();
        if (node_ordered.first) {
            double t1 = timestamp();
            // not_captured = default rule truthtable & ~ captured
            rule_vandnot(not_captured,
                         tree->rule(0).truthtable, captured,
                         tree->nsamples(), &cnt);
            evaluate_children(tree, node_ordered.first, node_ordered.second, not_captured, q, p);
            logger->addToEvalChildrenTime(time_diff(t1));
            logger->incEvalChildrenNum();

            if (tree->min_objective() < min_objective) {
                //printf("Hello!\n");
                min_objective = tree->min_objective();
                if (verbosity >= 10)
                    printf("before garbage_collect. num_nodes: %zu, log10(remaining): %zu\n", 
                            tree->num_nodes(), logger->getLogRemainingSpaceSize());
                logger->dumpState();
                tree->garbage_collect();
                logger->dumpState();
                if (verbosity >= 10)
                    printf("after garbage_collect. num_nodes: %zu, log10(remaining): %zu\n", tree->num_nodes(), logger->getLogRemainingSpaceSize());
                //printf("Goodbye!\n");
            }
        }
        logger->setQueueSize(q->size());
        if (queue_min_length < logger->getQueueMinLen()) {
            // garbage collect the permutation map: can be simplified for the case of BFS
            queue_min_length = logger->getQueueMinLen();
            //pmap_garbage_collect(p, queue_min_length);
        }
        ++num_iter;
        if ((num_iter % 10000) == 0) {
            if (verbosity >= 10)
                printf("iter: %zu, tree: %zu, queue: %zu, pmap: %zu, log10(remaining): %zu, time elapsed: %f\n",
                       num_iter, tree->num_nodes(), q->size(), p->size(), logger->getLogRemainingSpaceSize(), time_diff(start));
        }
        if ((num_iter % logger->getFrequency()) == 0) {
            // want ~1000 records for detailed figures
            logger->dumpState();
        }
    }
    logger->dumpState(); // second last log record (before queue elements deleted)
    if (verbosity >= 1)
        printf("iter: %zu, tree: %zu, queue: %zu, pmap: %zu, log10(remaining): %zu, time elapsed: %f\n",
               num_iter, tree->num_nodes(), q->size(), p->size(), logger->getLogRemainingSpaceSize(), time_diff(start));
    if (q->empty())
        printf("Exited because queue empty\n");
    else
        printf("Exited because max number of nodes in the tree was reached\n");

    size_t tree_mem = logger->getTreeMemory(); 
    size_t pmap_mem = logger->getPmapMemory(); 
    size_t queue_mem = logger->getQueueMemory(); 
    printf("TREE mem usage: %zu\n", tree_mem);
    printf("PMAP mem usage: %zu\n", pmap_mem);
    printf("QUEUE mem usage: %zu\n", queue_mem);

    // Print out queue
    ofstream f;
    if (print_queue) {
        char fname[] = "queue.txt";
        printf("Writing queue elements to: %s\n", fname);
        f.open(fname, ios::out | ios::trunc);
        f << "lower_bound objective length frac_captured rule_list\n";
    }

    // Clean up data structures
    printf("Deleting queue elements and corresponding nodes in the cache,"
            "since they may not be reachable by the tree's destructor\n");
    printf("\nminimum objective: %1.10f\n", tree->min_objective());
    Node* node;
    double min_lower_bound = 1.0;
    double lb;
    size_t num = 0;
    while (!q->empty()) {
        node = q->front();
        q->pop();
        if (node->deleted()) {
            tree->decrement_num_nodes();
            logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
            delete node;
        } else {
            lb = node->lower_bound() + tree->c();
            if (lb < min_lower_bound)
                min_lower_bound = lb;
            if (print_queue) {
                std::pair<tracking_vector<unsigned short, DataStruct::Tree>, tracking_vector<bool, DataStruct::Tree> > pp_pair = node->get_prefix_and_predictions();
                tracking_vector<unsigned short, DataStruct::Tree> prefix = std::move(pp_pair.first);
                tracking_vector<bool, DataStruct::Tree> predictions = std::move(pp_pair.second);
                f << node->lower_bound() << " " << node->objective() << " " << node->depth() << " "
                  << (double) node->num_captured() / (double) tree->nsamples() << " ";
                for(size_t i = 0; i < prefix.size(); ++i) {
                    f << tree->rule_features(prefix[i]) << "~"
                      << predictions[i] << ";";
                }
                f << "default~" << predictions.back() << "\n";
                num++;
            }
        }
    }
    printf("minimum lower bound in queue: %1.10f\n\n", min_lower_bound);
    if (print_queue)
        f.close();
    // last log record (before cache deleted)
    logger->dumpState();

    rule_vfree(&captured);
    rule_vfree(&not_captured);
    return num_iter;
}
