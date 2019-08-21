#include "queue.hh"
#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <time.h>
#define BUFSZ 512

/*
 * Logs statistics about the execution of the algorithm and dumps it to a file.
 * To turn off, pass verbosity <= 1
 */
NullLogger* logger;
int sensitiveAttrColumn = 1;
int unsensitiveAttrColumn = 0;
bool isSpecifiedUnpro = false;
float beta = 0.00;
double min_fairness_acceptable = 0;
int mode = 1; // Default 
int fairnessBoundPruneCnt = 0;
long totalFairTime = 0;
bool useUnfairnessLB = true;
int random_k_best = 1;
int BFS_search_strat = 2;
int seed = 1;
double reason = 2;
int beginN = 1000;
bool performMS = true;
/*(1 -> beta tradeoff accuracy/fairness ; 
2 -> Find max fairness ; 
3 -> Find max acc given minimal acceptable fairness)
4 -> Find max accuracy */
int main(int argc, char *argv[]) {
    srand(seed);
    long int timemillis0 = clock()/CLOCKS_PER_SEC;
    const char usage[] = "USAGE: %s [-b] "
        "[-n max_num_nodes] [-r regularization] [-x sensitive_attribute_column] [-z beta] [-v verbosity] "
        "-c (1|2|3|4) -p (0|1|2) [-f logging_frequency]"
        "-a (0|1|2) [-s] [-L latex_out] [-m 1|2|3|4] [-h min_fairness (si -m 3)] [-g unsensitive_attribute_column]"
        "[-i (1|2)] [-k nb_k_best] [-d 0|1|2|3] [-e [-j begin_limit_nodes] [-o reason]]"
        "data.out data.label\n\n"
        "%s\n";

    extern char *optarg;
    bool run_bfs = false;
    bool run_curiosity = false;
    int curiosity_policy = 0;
    bool latex_out = false;
    bool use_prefix_perm_map = false;
    bool use_captured_sym_map = false;
    int verbosity = 0;
    int map_type = 0;
    int max_num_nodes = 100000;
    double c = 0.01;
    char ch;
    bool error = false;
    char error_txt[BUFSZ];
    int freq = 1000;
    int ablation = 0;
    bool calculate_size = false;
    int useLB;
    /* only parsing happens here */
    while ((ch = getopt(argc, argv, "bsLec:p:v:n:r:f:a:x:z:m:h:g:i:k:d:j:o:")) != -1) {
        switch (ch) {
        case 'b':
            run_bfs = true;
            break;
        case 'm':
            mode = atoi(optarg);
            break;
        case 'i':
            useLB = atoi(optarg);
            if(useLB == 0) {
                useUnfairnessLB = false;
            } else if(useLB == 1) {
                useUnfairnessLB = true;
            }
            else {
                error = true;
            }
            break;
        case 'h':
            if(mode != 3) {
                error = true;
            } else {
                min_fairness_acceptable = atof(optarg);
            }
            break;
        case 's':
            calculate_size = true;
            break;
        case 'c':
            run_curiosity = true;
            curiosity_policy = atoi(optarg);
            break;
        case 'L':
            latex_out = true;
            break;
        case 'p':
            map_type = atoi(optarg);
            use_prefix_perm_map = map_type == 1;
            use_captured_sym_map = map_type == 2;
            break;
        case 'v':
            verbosity = atoi(optarg);
            break;
        case 'n':
            max_num_nodes = atoi(optarg);
            break;
        case 'r':
            c = atof(optarg);
            break;
        case 'f':
            freq = atoi(optarg);
            break;
        case 'a':
            ablation = atoi(optarg);
            break;
        case 'x':
            sensitiveAttrColumn = atoi(optarg);
            break;
        case 'g':
            unsensitiveAttrColumn = atoi(optarg);
            isSpecifiedUnpro = true;
            break;
        case 'z':
            beta = atof(optarg);
            break;
        case 'k':
            random_k_best = atoi(optarg);
            break;
        case 'd':
            BFS_search_strat = atoi(optarg);
            if(BFS_search_strat > 3 || BFS_search_strat < 0) {
                error = true;
            }
            break;
        case 'e':
            performMS = true;
            break;
        case 'j':
            beginN = atoi(optarg);
            if(beginN <= 0) {
                error = true;
            }
            break;
        case 'o':
            reason = atof(optarg);
            if(reason < 1) {
                error = true;
            }
            break;
        default:
            error = true;
            snprintf(error_txt, BUFSZ, "unknown option: %c", ch);
        }
    }
    if(mode == 2) { // bounds do not hold when optimizing only fairness !
        ablation = 3;
    }
    if (max_num_nodes < 0) {
        error = true;
        snprintf(error_txt, BUFSZ, "number of nodes must be positive");
    }
    if (c < 0) {
        error = true;
        snprintf(error_txt, BUFSZ, "regularization constant must be postitive");
    }
    if (map_type > 2 || map_type < 0) {
        error = true;
        snprintf(error_txt, BUFSZ, "symmetry-aware map must be (0|1|2)");
    }
    if ((run_bfs + run_curiosity) != 1) {
        error = true;
        snprintf(error_txt, BUFSZ,
                "you must use at least and at most one of (-b | -c)");
    }
    if (argc < 2 + optind) {
        error = true;
        snprintf(error_txt, BUFSZ,
                "you must specify data files for rules and labels");
    }
    if (run_curiosity && !((curiosity_policy >= 1) && (curiosity_policy <= 4))) {
        error = true;
        snprintf(error_txt, BUFSZ,
                "you must specify a curiosity type (1|2|3|4)");
    }

    if (error) {
        fprintf(stderr, usage, argv[0], error_txt);
        exit(1);
    }

    if(!performMS) {
        std::map<int, std::string> curiosity_map;
        curiosity_map[1] = "curiosity";
        curiosity_map[2] = "curious_lb";
        curiosity_map[3] = "curious_obj";
        curiosity_map[4] = "dfs";

        argc -= optind;
        argv += optind;

        int nrules, nsamples, nlabels, nsamples_chk;
        rule_t *rules, *labels;
        rules_init(argv[0], &nrules, &nsamples, &rules, 1);
        rules_init(argv[1], &nlabels, &nsamples_chk, &labels, 0);

        int nmeta, nsamples_check;
        // Equivalent points information is precomputed, read in from file, and stored in meta
        rule_t *meta;
        if (argc == 3)
            rules_init(argv[2], &nmeta, &nsamples_check, &meta, 0);
        else
            meta = NULL;

        if (verbosity >= 10)
            print_machine_info();
        char froot[BUFSZ];
        char log_fname[BUFSZ];
        char opt_fname[BUFSZ];
        const char* pch = strrchr(argv[0], '/');
        snprintf(froot, BUFSZ, "../logs/for-%s-%s%s-%s-%s-removed=%s-max_num_nodes=%d-c=%.7f-v=%d-f=%d",
                pch ? pch + 1 : "",
                run_bfs ? "bfs" : "",
                run_curiosity ? curiosity_map[curiosity_policy].c_str() : "",
                use_prefix_perm_map ? "with_prefix_perm_map" : 
                    (use_captured_sym_map ? "with_captured_symmetry_map" : "no_pmap"),
                meta ? "minor" : "no_minor",
                ablation ? ((ablation == 1) ? "support" : "lookahead") : "none",
                max_num_nodes, c, verbosity, freq);
        snprintf(log_fname, BUFSZ, "%s.txt", froot);
        snprintf(opt_fname, BUFSZ, "%s-opt.txt", froot);

        if (verbosity >= 1000) {
            printf("\n%d rules %d samples\n\n", nrules, nsamples);
            rule_print_all(rules, nrules, nsamples);

            printf("\nLabels (%d) for %d samples\n\n", nlabels, nsamples);
            rule_print_all(labels, nlabels, nsamples);
        }

        if (verbosity > 1)
            logger = new Logger(c, nrules, verbosity, log_fname, freq);
        else
            logger = new NullLogger();
        double init = timestamp();
        char run_type[BUFSZ];
        Queue* q;
        strcpy(run_type, "LEARNING RULE LIST via ");
        char const *type = "node";
        if (curiosity_policy == 1) {
            strcat(run_type, "CURIOUS");
            q = new Queue(curious_cmp, run_type);
            type = "curious";
        } else if (curiosity_policy == 2) {
            strcat(run_type, "LOWER BOUND");
            q = new Queue(lb_cmp, run_type);
        } else if (curiosity_policy == 3) {
            strcat(run_type, "OBJECTIVE");
            q = new Queue(objective_cmp, run_type);
        } else if (curiosity_policy == 4) {
            strcat(run_type, "DFS");
            q = new Queue(dfs_cmp, run_type);
        } else {
            strcat(run_type, "BFS");
            q = new Queue(base_cmp, run_type);
        }

        PermutationMap* p;
        if (use_prefix_perm_map) {
            strcat(run_type, " Prefix Map\n");
            PrefixPermutationMap* prefix_pmap = new PrefixPermutationMap;
            p = (PermutationMap*) prefix_pmap;
        } else if (use_captured_sym_map) {
            strcat(run_type, " Captured Symmetry Map\n");
            CapturedPermutationMap* cap_pmap = new CapturedPermutationMap;
            p = (PermutationMap*) cap_pmap;
        } else {
            strcat(run_type, " No Permutation Map\n");
            NullPermutationMap* null_pmap = new NullPermutationMap;
            p = (PermutationMap*) null_pmap;
        }

        CacheTree* tree = new CacheTree(nsamples, nrules, c, rules, labels, meta, ablation, calculate_size, type);
        printf("%s", run_type);
        // runs our algorithm
        bbound(tree, max_num_nodes, q, p);
        if (mode == 3) {
            printf("total branches ignored with fairness upper bound : %d\n", fairnessBoundPruneCnt);
        }
        long int timemillis1 = clock()/CLOCKS_PER_SEC;
        printf("Total time spent for fairness calculations : %d s\n", totalFairTime);
        printf("Total execution time : %d s\n", timemillis1 - timemillis0);
        printf("final num_nodes: %zu\n", tree->num_nodes());
        printf("final num_evaluated: %zu\n", tree->num_evaluated());
        printf("final min_objective: %1.5f\n", tree->min_objective());
        const tracking_vector<unsigned short, DataStruct::Tree>& r_list = tree->opt_rulelist();
        accAndFair metricsA = computeFinalFairness(nsamples, r_list, tree->opt_predictions(),
                        rules, labels, isSpecifiedUnpro, unsensitiveAttrColumn, sensitiveAttrColumn);
        /*printf("final accuracy: %1.5f\n",
        1 - tree->min_objective() + c*r_list.size());*/
        printf("final accuracy: %lf\n",
        metricsA.accuracy);
        printf("final statistical parity: %lf\n", metricsA.fairness);
        print_final_rulelist(r_list, tree->opt_predictions(),
                        latex_out, rules, labels, opt_fname);
        printf("final total time: %f\n", time_diff(init));
        logger->dumpState();
        logger->closeFile();
        if (meta) {
            printf("\ndelete identical points indicator");
            rules_free(meta, nmeta, 0);
        }
        printf("\ndelete rules\n");
        rules_free(rules, nrules, 1);
        printf("delete labels\n");
        rules_free(labels, nlabels, 0);
        printf("tree destructors\n");
        return 0;
    } else {
        printf("Will perform restart with randomization\n");
        printf("Starting with %d maximum nodes ; reason = %lf ; final max number = %d\n", beginN, reason, max_num_nodes);
        BFS_search_strat = 4; // else it is useless to perform restart !
        /* ------------- COMMON INFO NEEDED FOR FINAL PRINT AND COMP -------------- */
        int nmeta, nsamples_check;
        // Equivalent points information is precomputed, read in from file, and stored in meta
        rule_t *meta;
        std::map<int, std::string> curiosity_map;
        curiosity_map[1] = "curiosity";
        curiosity_map[2] = "curious_lb";
        curiosity_map[3] = "curious_obj";
        curiosity_map[4] = "dfs";
        
        argc -= optind;
        argv += optind;

        if (argc == 3)
            rules_init(argv[2], &nmeta, &nsamples_check, &meta, 0);
        else
            meta = NULL;
        int nrules, nsamples, nlabels, nsamples_chk;
        rule_t *rules, *labels;
        rules_init(argv[0], &nrules, &nsamples, &rules, 1);
        rules_init(argv[1], &nlabels, &nsamples_chk, &labels, 0);
        char froot[BUFSZ];
        char log_fname[BUFSZ];
        char opt_fname[BUFSZ];
        const char* pch = strrchr(argv[0], '/');
        double init = timestamp();
        snprintf(froot, BUFSZ, "../logs/for-%s-%s%s-%s-%s-removed=%s-max_num_nodes=%d-c=%.7f-v=%d-f=%d", 
                pch ? pch + 1 : "",
                run_bfs ? "bfs" : "",
                run_curiosity ? curiosity_map[curiosity_policy].c_str() : "",
                use_prefix_perm_map ? "with_prefix_perm_map" : 
                    (use_captured_sym_map ? "with_captured_symmetry_map" : "no_pmap"),
                meta ? "minor" : "no_minor",
                ablation ? ((ablation == 1) ? "support" : "lookahead") : "none",
                max_num_nodes, c, verbosity, freq);
        snprintf(log_fname, BUFSZ, "%s.txt", froot);
        snprintf(opt_fname, BUFSZ, "%s-opt.txt", froot);
        /* ---------------------- POINTERS THAT WILL POINT TO BEST SOLUTION ------------- */
        CacheTree* bestTree;
        // compute number of executions
        int test = beginN;
        int nb = 0;
        while(max_num_nodes > test) {
            test = reason * test;
            nb++;
        }
        bool first = true;
        int currLimit = beginN;
        while(max_num_nodes > currLimit) {
            printf("Current nodes limit = %d\n", currLimit);
            argc -= optind;
            argv += optind;

            Queue* q;
            if (verbosity > 1)
                logger = new Logger(c, nrules, verbosity, log_fname, freq);
            else
                logger = new NullLogger();
            char run_type[BUFSZ];
            
            strcpy(run_type, "LEARNING RULE LIST via ");
            char const *type = "node";
            if (curiosity_policy == 1) {
                strcat(run_type, "CURIOUS");
                q = new Queue(curious_cmp, run_type);
                type = "curious";
            } else if (curiosity_policy == 2) {
                strcat(run_type, "LOWER BOUND");
                q = new Queue(lb_cmp, run_type);
            } else if (curiosity_policy == 3) {
                strcat(run_type, "OBJECTIVE");
                q = new Queue(objective_cmp, run_type);
            } else if (curiosity_policy == 4) {
                strcat(run_type, "DFS");
                q = new Queue(dfs_cmp, run_type);
            } else {
                strcat(run_type, "BFS");
                q = new Queue(base_cmp, run_type);
            }

            PermutationMap* p;
            if (use_prefix_perm_map) {
                strcat(run_type, " Prefix Map\n");
                PrefixPermutationMap* prefix_pmap = new PrefixPermutationMap;
                p = (PermutationMap*) prefix_pmap;
            } else if (use_captured_sym_map) {
                strcat(run_type, " Captured Symmetry Map\n");
                CapturedPermutationMap* cap_pmap = new CapturedPermutationMap;
                p = (PermutationMap*) cap_pmap;
            } else {
                strcat(run_type, " No Permutation Map\n");
                NullPermutationMap* null_pmap = new NullPermutationMap;
                p = (PermutationMap*) null_pmap;
            }
            CacheTree* tree = new CacheTree(nsamples, nrules, c, rules, labels, meta, ablation, calculate_size, type);
            
            //printf("%s", run_type);
            // runs our algorithm
            bbound(tree, currLimit, q, p);
            logger->dumpState();
            logger->closeFile();
            
            currLimit*=reason;
            printf("best objective function value reached = %lf\n",tree->min_objective());
            if(first) {
                bestTree = tree;
                first = false;
            } else {
                if(tree->min_objective() < bestTree->min_objective()) {
                    bestTree = tree;
                }
            }
        }
        long int timemillis1 = clock()/CLOCKS_PER_SEC;
        printf("Total execution time : %d s\n", timemillis1 - timemillis0);
        printf("final num_nodes: %zu\n", bestTree->num_nodes());
        printf("final num_evaluated: %zu\n", bestTree->num_evaluated());
        printf("final min_objective: %1.5f\n", bestTree->min_objective());
        const tracking_vector<unsigned short, DataStruct::Tree>& r_list = bestTree->opt_rulelist();
        accAndFair metricsA = computeFinalFairness(nsamples, r_list, bestTree->opt_predictions(),
                        rules, labels, isSpecifiedUnpro, unsensitiveAttrColumn, sensitiveAttrColumn);
        /*printf("final accuracy: %1.5f\n",
        1 - tree->min_objective() + c*r_list.size());*/
        printf("final accuracy: %lf\n",
        metricsA.accuracy);
        printf("final statistical parity: %lf\n", metricsA.fairness);
        print_final_rulelist(r_list, bestTree->opt_predictions(),
                        latex_out, rules, labels, opt_fname);
        printf("final total time: %f\n", time_diff(init));
        
        if (meta) {
            printf("\ndelete identical points indicator");
            rules_free(meta, nmeta, 0);
        }
        printf("\ndelete rules\n");
        rules_free(rules, nrules, 1);
        printf("delete labels\n");
        rules_free(labels, nlabels, 0);
        printf("tree destructors\n");
        return 0;
    }
}
