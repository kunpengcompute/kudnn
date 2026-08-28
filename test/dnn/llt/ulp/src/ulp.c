/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuDNN is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "com_testcase_ulp.h"
#include "random.h"
#include <getopt.h>

#ifndef MAX_ARGS_NUM
#define MAX_ARGS_NUM 6
#endif

typedef void (*FuncPtr)(void);
typedef int (*Args0Func)(struct UlpInfo *ulpInfo);
typedef int (*Args1Func)(const char* arg, struct UlpInfo *ulpInfo);

typedef enum PRASE_ARGS_TYPE {
    PRASE_ARGS0,
    PRASE_ARGS0_SPECIAL,
    PRASE_ARGS1,
    PRASE_ARGS2,
} PRASE_ARGS_TYPE;

struct ParseCmdTabItem {
    char opt;
    PRASE_ARGS_TYPE type;
    FuncPtr func;
};

static void usage()
{
    puts("./ulp [-s] [-q] [-g] [-r nudz] [-mode la] [-l vector_len] [-e ulplimit] [-b errorBase] func lo [hi [x lo2 hi2] [count]]");
    puts("Compares func against a higher precision implementation in [lo; hi].");
    puts("-q: enable print ulp not 0.");
    puts("-m: use mpfr even if faster method is available.");
    puts("-l: set vector len for vml function test.");
    puts("-g: enable gdb printf into file, printf ulp != 0 and result.");
    puts("-c: enable call in-place interface");
    puts("-r: nudz(defaut:FE_TONEAREST) can set n:FE_TONEAREST, u:FE_UPWARD, d:FE_DOWNWARD, z:FE_TOWARDZERO");
    puts("-mode: (defaut:KPL_LA) can set la:KPL_LA, ha:KPL_HA, ep:KPL_EP");
    puts("-s: set random seed for generation cases.");
    puts("-e: set ulp or relative error limit.");
    puts("-t: set linear or random testcase generate strategy.");
    puts("-b: (defaut:0) set relative errorBase.");
    puts("    1) errorBase=0 use ulp check");
    puts("    2) errorBase!=0 and res>errorBase use relative error check");
    puts("    3) errorBase!=0 and res<errorBase use absolute error check");

    exit(1);
}

static int InitQuiet(struct UlpInfo *ulpInfo)
{
    int ret = 0;
    ulpInfo->quiet = 1;
    return ret;
}

static int InitGdbPrint(struct UlpInfo *ulpInfo)
{
    int ret = 0;
    ulpInfo->gdbPrint = 1;
    return ret;
}

static int SetExecuteMode(struct UlpInfo *ulpInfo)
{
    int ret = 0;
    ulpInfo->exeMode = IN_PLACE;
    return ret;
}

static int InitIntervalValue(struct UlpInfo *ulpInfo)
{
    int ret = 0;
    return ret;
}

static int InitSeed(const char *val, struct UlpInfo *ulpInfo)
{
    int ret = 0;
    ulpInfo->seed = strtoul(val, 0, 0);
    return ret;
}

static int InitRoundMode(const char *val, struct UlpInfo *ulpInfo)
{
    int ret = 0;
    ulpInfo->round = FE_TONEAREST;
    return ret;
}

static int InitAccuracyMode(const char *val, struct UlpInfo *ulpInfo)
{
    int ret = 0;
    if (strcmp(val, "la") == 0) {
        ulpInfo->mode = KPL_LA;
    } else if (strcmp(val, "ha") == 0) {
        ulpInfo->mode = KPL_HA;
    } else if (strcmp(val, "ep") == 0) {
        ulpInfo->mode = KPL_EP;
    } else {
        usage();
        ret = 1;
    }
    ulpInfo->modeFlag = 1;
    return ret;
}

static int InitVectorLen(const char *val, struct UlpInfo *ulpInfo)
{
    int ret = 0;
    ulpInfo->lengthDnn = strtol(val, 0, 0);
    ulpInfo->length = strtol(val, 0, 0);
    return ret;
}

static int InitUlpThreshold(const char *val, struct UlpInfo *ulpInfo)
{
    int ret = 0;
    ulpInfo->errorlim = strtod(val, 0);
    return ret;
}

static int InitErrorBase(const char *val, struct UlpInfo *ulpInfo)
{
    int ret = 0;
    ulpInfo->errorBase = strtod(val, 0);
    return ret;
}

static int InitTestMode(const char *val, struct UlpInfo *ulpinfo)
{
    int ret = 0;
    if (!strcmp(val, "l") || !strcmp(val, "linear")) {
        ulpinfo->testMode = TMODE_LINEAR;
    } else if (!strcmp(val, "r") || !strcmp(val, "random") || !strcmp(val, "rand")) {
        ulpinfo->testMode = TMODE_RANDOM;
    } else {
        usage();
        return -1;
    }
    return ret;
}

static int UlpParse(int argc, char **argv, struct UlpInfo *ulpInfo)
{
    /* Init ulpInfo struct */
    ulpInfo->seed = GetRandomU64();
    ulpInfo->round = FE_TONEAREST;
    ulpInfo->modeFlag = 0;
    ulpInfo->length = 1023;
    ulpInfo->lengthDnn = 1023;
    ulpInfo->errorlim = INFINITY;
    ulpInfo->gdbPrint = 0;
    ulpInfo->quiet = 0;
    ulpInfo->times = 1;

    /* parse command line [-q] [-g] [-s seed] [-r nudz] [-mode la] [-l vector_len] [-e ulplimit] [-b errorBase] parameter */
    int ret = 0;
    const static struct option longOpts[] = {
        {"mode", required_argument, NULL, 'm'},
        {"errorBase", required_argument, NULL, 'b'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    const char *string = "cpqghs:r:m:e:l:b:t:";
    int opt;
    int parseFlag = 0;
    const static struct ParseCmdTabItem parseCmdTab[] = {
        {'q', PRASE_ARGS0, (FuncPtr)InitQuiet},
        {'g', PRASE_ARGS0, (FuncPtr)InitGdbPrint},
        {'c', PRASE_ARGS0, (FuncPtr)SetExecuteMode},
        {'p', PRASE_ARGS0, (FuncPtr)InitIntervalValue},
        {'s', PRASE_ARGS1, (FuncPtr)InitSeed},
        {'r', PRASE_ARGS1, (FuncPtr)InitRoundMode},
        {'m', PRASE_ARGS1, (FuncPtr)InitAccuracyMode},
        {'e', PRASE_ARGS1, (FuncPtr)InitUlpThreshold},
        {'t', PRASE_ARGS1, (FuncPtr)InitTestMode},
        {'l', PRASE_ARGS1, (FuncPtr)InitVectorLen},
        {'b', PRASE_ARGS1, (FuncPtr)InitErrorBase},
        {'h', PRASE_ARGS0_SPECIAL, (FuncPtr)usage},
    };
    while ((opt = getopt_long_only(argc, argv, string, longOpts, NULL)) != -1) {
        for (int i = 0; i < sizeof(parseCmdTab) / sizeof(struct ParseCmdTabItem); i++) {
            if (opt == parseCmdTab[i].opt) {
                if (PRASE_ARGS1 == parseCmdTab[i].type) {
                    ret = ((Args1Func)(parseCmdTab[i].func))(optarg, ulpInfo);
                    parseFlag += 2;
                } else if (PRASE_ARGS0 == parseCmdTab[i].type) {
                    ret = ((Args0Func)(parseCmdTab[i].func))(ulpInfo);
                    parseFlag++;
                } else if (PRASE_ARGS0_SPECIAL == parseCmdTab[i].type) {
                    ((parseCmdTab[i].func))();
                }
                break;
            }
            if (ret) {
                return ret;
            }
        }

        if (argv[1 + parseFlag][0] != '-') {
            break;
        }
    }

    /* parse function name */
    argc -= parseFlag;
    argv += parseFlag;
    argc--;
    argv++;
    const struct FunctionInfo *f = NULL;
    for (f = g_functions; f->ulp_name; f++) {
        if (strcmp(argv[0], f->ulp_name) == 0) {
            break;
        }
    }
    if (!f->ulp_name) {
        printf("tool not support %s\n", argv[0]);
        usage();
        ret = 1;
    }
    ulpInfo->funcInfo = f;

    /* parse cases interval and test times */
    argc--;
    argv++;
    if (argc < 1) {
        usage();
        ret = 1;
    }
    for (int i = 0; i < nulpcases; i++) {
        if (ulpInfo->funcInfo->funcType == g_ulpcases[i].funcType) {
            ((UlpParseFuncPtr)g_ulpcases[i].parse)(argc, argv, ulpInfo);
            break;
        }
    }

    return ret;
}

int main(int argc, char *argv[])
{
    /* 1.prase input and store informtion into ulpInfo struct */
    struct UlpInfo *ulpInfo = (struct UlpInfo *)calloc(1, sizeof(struct UlpInfo));
    UlpParse(argc, argv, ulpInfo);

    /* plant random seed */
    printf("seed : %#lx\n", ulpInfo->seed);
    seed_random(ulpInfo->seed);

    FILE *fp = NULL;
    if (ulpInfo->gdbPrint) {
        fp = freopen("ulp_fail_arm.log","w",stdout);
        if(NULL == fp) {
            printf("reopen failed\n");
            return -1;
        }
    }

    /* 2-6. calculate ulp error */
    for (int i = 0; i < nulpcases; i++) {
        if (ulpInfo->funcInfo->funcType == g_ulpcases[i].funcType) {
            ((UlpCaseFuncPtr)g_ulpcases[i].func)(ulpInfo);
            break;
        }
    }

    if (ulpInfo->gdbPrint) {
        fclose(fp);
    }
    free(ulpInfo);
}
