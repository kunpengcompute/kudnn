/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuDNN is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may   obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <kupl.h>

#define CONSTRUCTOR __attribute__((constructor))
#define DESTRUCTOR __attribute__((destructor))
#define PRIVATE __attribute__((visibility("hidden")))
PRIVATE void CONSTRUCTOR KudnnInit(void);
PRIVATE void DESTRUCTOR KudnnShutDown(void);

#define ARM_CPU_IMP_HISI 0x48

static bool KudnnHwDetect(void)
{
    unsigned long long cpuId;
    __asm__ volatile("mrs %0, MIDR_EL1" : "=r"(cpuId));

    unsigned long long vendor = (cpuId >> 0x18) & 0xFF;
    if (vendor == ARM_CPU_IMP_HISI) {
        return true;
    }
    return false;
}

#ifdef __cplusplus
extern "C" {
#endif

extern kupl_egroup_h *g_egroups;
kupl_egroup_h *g_egroups = NULL;

static int KuplInit(void)
{
    const char *env = getenv("KUDNN_THREAD_TYPE");

    if ((env != NULL) && strcmp(env, "KUPL") == 0) {
        int enums = kupl_get_num_executors();
        g_egroups = (kupl_egroup_h *)calloc(enums, sizeof(kupl_egroup_h));
        if (g_egroups == NULL) {
            fprintf(stderr, "KML_BLAS: malloc failed in %s\n", __func__);
            return -1;
        }
    }
    return 0;
}

static void KuplEgroupsDestory(void)
{
    if (g_egroups == NULL) {
        return;
    }
    for (int i = 0; i < kupl_get_num_executors(); i++) {
        kupl_egroup_destroy(g_egroups[i]);
    }
    free(g_egroups);
    g_egroups = NULL;
}
#ifdef __cplusplus
}
#endif

PRIVATE void CONSTRUCTOR KudnnInit(void)
{
    if (!KudnnHwDetect()) {
        fprintf(stderr, "KuDNN: The library is running into an error, please check CPU architect.\n");
        exit(EXIT_FAILURE);
    }
    if (KuplInit()) {
        fprintf(stderr, "KuDNN: KuplInit failed.\n");
        exit(EXIT_FAILURE);
    }
    return;
}

PRIVATE void DESTRUCTOR KudnnShutDown(void)
{
    KuplEgroupsDestory();
}
