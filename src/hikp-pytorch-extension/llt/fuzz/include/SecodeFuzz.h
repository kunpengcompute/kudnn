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
 
#ifndef __MUTATOR_PUBLIC_H__
#define __MUTATOR_PUBLIC_H__
 
#ifndef __KERNEL__
 
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h> // sqrt cos log
 
 
#ifndef _WIN32
 
#include <sys/time.h> // struct itimeral. setitimer()
#include <sys/wait.h>
#include <unistd.h>
#include <sys/resource.h>
#include <pthread.h>
 
#endif
 
#define __TIME_ENABLE__
#ifdef __TIME_ENABLE__
 
#include <time.h> // time
 
#endif
 
#else
#include <linux/module.h>
#include <linux/slab.h> /* kmalloc() */
#endif
 
// ====================== 宏开关 ======================
#define LIB_DEFINE // 系统库宏定义
#define HAS_IO     // 文件读写
#define HAS_HOOK
#define HAS_TRACE_PC
#define HAS_LEAK_CHECK
#define HAS_CMD
#define HAS_SIGNAL
/* 内核覆盖反馈模块
#define HAS_KCOV */
 
/* 多线程支持仅仅为了发现多线程访问引起的问题
 不能为了提高测试速度使用它，提高测试速度请使用多进程
 大多数情况测试不要打开它
 #define SUPPORT_M_THREAD */
 
// vs编译器要裁剪的
#ifdef _MSC_VER
 
#ifndef __clang__
#undef HAS_HOOK
#undef HAS_TRACE_PC
#undef HAS_LEAK_CHECK
#undef HAS_SIGNAL
#endif
#undef HAS_KCOV
#endif
 
// 编进内核态，大多裁剪了
#ifdef __KERNEL__
#undef LIB_DEFINE
#undef HAS_IO
#undef HAS_HOOK
#undef HAS_TRACE_PC
#undef HAS_LEAK_CHECK
#undef HAS_CMD
#undef HAS_SIGNAL
#undef HAS_KCOV
#endif
 
/* *****************************************
 
定义一些规格
 
***************************************** */
 
// #define little_mem            1                   // 内存非常非常吃紧的时候使用这个宏
#ifdef little_mem
 
#undef LIB_DEFINE
#undef HAS_IO
#undef HAS_HOOK
#undef HAS_TRACE_PC
#undef HAS_LEAK_CHECK
#undef HAS_CMD
#undef HAS_SIGNAL
#undef HAS_KCOV
#endif
 
#ifdef HAS_KCOV
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif
 
#ifdef HAS_SIGNAL
 
#include <signal.h>
#include "errno.h"
 
#endif
 
/* 针对没有malloc系统的，一般别打开(使用时可能还要具体调整其他宏)
 #define malloc_self  1 */
 
#ifdef __cplusplus
extern "C" {
#endif
 
/* *****************************************
 
整数类型声明
 
***************************************** */
typedef uint8_t sec_u8;
typedef uint16_t sec_u16;
typedef uint32_t sec_u32;
typedef uint64_t sec_u64;
typedef int8_t sec_s8;
typedef int16_t sec_s16;
typedef int32_t sec_s32;
typedef int64_t sec_s64;
 
 
/* *****************************************
 
结构体声明
 
***************************************** */
#define MAX_Name_Len 32
#define MAX_mutater_Len 32
#define enum_MutatedMAX 100
 
 
typedef struct {
    char name[MAX_Name_Len];
    char mutaterName[MAX_mutater_Len];
    int type;
    int inType;
    int len;
    int maxLen;
    char *value;
    char *desc;
} SPara;
 
typedef struct {
    SPara para;
    int isHasInitValue; // 是否有初始值
    int isNeedFree;
    int isAddWholeRandom; // 这个用来干啥
    int mutatedFrequency;
    int isAddWholeSequence;
    int sequenceStartPos;
 
    char numberValue[8]; // 专门用来保存数字类型初始值的，省的单独分配内存
 
    // sec_s64--sec_s32 修改看看
    sec_s32 inLen; // 数据的长度，以bit为单位，string必须被8整除,blob也是
    char *inBuf;
    int pos; // 目前执行的测试用例
 
    // 输出的数据，只读
    int count;            // 一共多少测试例
    int isNeedFreeOutBuf; // 释放outBuf标志
    int length;
    int sign; // 0=无符号，有符号
 
    // 用户可以改变
    int isMutatedClose[enum_MutatedMAX]; // 可以针对某个元素单独关闭变异算法，
 
    // 内部变量，请别动
    int isMutatedSupport[enum_MutatedMAX];
    int posStart[enum_MutatedMAX];
    int num[enum_MutatedMAX];
    int numbak[enum_MutatedMAX];
 
    int *enumNumberTable;
    char **enumStringTable;
    char **enumBlobTable;
    int *enumBloblTable;
    int enumCount;
 
    int min;
    int max;
 
    int arg;
 
    void *init; // SElementInit
} SElement;
 
typedef struct {
    int first;
    int type;
    int inType;
    int isHasInitValue;
    int isNeedFree;
    char *initValueBuffer;
    char *desc;
    sec_u64 initValue;
 
    int len; // number单位bit   other单位byte
    int maxLen;
    // 用于结构体
    void *structPtr;
    int structStart;
    int structLength;
 
    int *enumNumberTable;
    char **enumStringTable;
    char **enumBlobTable;
    int *enumBloblTable;
    int enumCount;
 
    int min;
    int max;
 
    int arg;
 
    SElement *element;
} SElementInit;
 
/* ***********************************************************************
 
0.一些辅助功能接口
 
*********************************************************************** */
// 0.设置是否在crash时打印crash样本在屏幕上,默认打印，1为打印，0为不打印
extern void DT_Set_If_Show_crash(int isShowAll);
 
// 1.设置是否在测试例结束时打印样本在屏幕上,默认不打印，1为打印，0为不打印
extern void DT_Set_If_Show_Corpus(int isShowAll);
 
// 2.设置打印测试例执行报告的路径，如果为空则不打印测试例报告(报告名后边自动补充时间后缀)
// path              = "./report"    当前目录下生成"report_时间戳"报告
// path              = "./report/"   report目录下生成"时间戳"报告
extern void DT_Set_Report_Path(const char *path);
 
// 2.1.设置打印测试例执行报告的文件名称，如果为空则不打印测试例报告(文件原先有内容则新测试追加到文件末尾)
// 与上一个函数二选一，谁后设置谁生效
// pathName              = "./report"       当前目录下生成"report"报告
// pathName              = "./report/log"   report目录下生成"log"报告
extern void DT_Set_Report_Fix_PathName(const char *pathName);
 
// 2.2 使能输出xml报告，将所有fuzz report输出到一个xml里。因文件太大，默认不生成xml
// isEnable  0 不生成xml  1 生成xml
extern void DT_SetEnableXmlReport(int isEnable);
 
// 3.设置是否输出覆盖率sancov文件，在clang编译器下有效，需要同时打开环境变量export ASAN_OPTIONS=coverage=1
// 默认不输出，1为输出，0为不输出
extern void DT_Set_Is_Dump_Coverage(int isDumpCoverage);
 
// 4.想要调整以下设置，请在最开始调用一次，目前请不要多次调用
// 调整后，使用的内存随之变化
// 设置最大输出尺寸，单位byte, 默认值为10000000
extern void DT_Set_MaxOutputSize(int imaxOutputSize);
 
// 设置最大参数个数，默认256  设为特别大会产生内存爆炸
extern void DT_Set_MaxParaNumber(int imaxParaNumber);
 
// 设置最大样本数量，默认是5000，最小是1
extern void DT_Set_MCorpusNm(int num);
 
// 设置样本进化支持的最大代码块数，默认2<<21
extern void DT_Set_MaxPcNumber(int num);
 
 
// 5.得到secodefuzz的版本信息
extern const char *DT_Get_Version(void);
 
// 6.设置单个测试例执行一次超过多长时间报bug,如果为0，则不检测超时bug,默认600秒
extern void DT_Set_TimeOut_Second(int second);
 
// 7.将二进制样本转换成文本字符串形式并打印,参数为二进制文件路径
// 输入比如为a.bin
extern void DT_Printf_Bin_To_Corpus(const char *path);
 
// 8.将二进制样本转换成文本字符串形式并写到文件里,参数为二进制文件路径
// 输入比如为b.bin
extern void DT_Write_Bin_To_Corpus(const char *path);
 
// 9.将样本字符串转换为二进制文件，并写到文件里，,参数为样本文件路径
// 输入为hexvalue ="\x00\x00\x00\x00\x00"样子的文件c.txt
extern void DT_Write_Corpus_To_Bin(const char *path);
 
// 10.设置是否开始或者停止监控覆盖反馈，一般不需要调用
extern void DT_Enable_TracePC(int isEnable);
 
// 11.主动打印当前样本和写入文件，利用函数返回值判断错误的时候使用
extern void DT_Show_Cur_Corpus(void);
 
// 12.控制debug开关函数，默认关闭，1为开启，0为关闭
extern void DT_Enable_Log(int isEnable);
 
// 13.得到变异数据的长度
extern int DT_GET_MutatedValueLen(SElementInit *init);
 
// 13.0设置变异数据变异频率百分比，100为%100,1为1%,-1为不变异，0为默认全局变异
// 即使为%100,因为有些变异算法的原因，也有少量情况可能没有变异
extern void DT_SET_MutatedFrequency(SElementInit *init, int frequency);
 
// 13.1得到某个参数是否被变异
extern int DT_GET_IsBeMutated(SElementInit *init);
 
// 14.关闭所有变异算法，默认全部开启，1为开启，0为关闭
extern void DT_Enable_AllMutater(int isEnable);
 
// 15.关闭某个变异算法，默认开启，1为开启，0为关闭
extern void DT_Enable_OneMutater(int mutatedNum, int isEnable);
 
// 16.设置字符串变异是否最后添加\0 (如果长度非0),默认为添加
extern void DT_Set_String_Has_Terminal(int isHasTerminal);
 
// 16.1设置字符串变异是否有Null
extern void DT_Set_String_Has_Null(int isHasNull);
 
// 17.设置第一次运行到代码块，打印块pc指针到屏幕，默认关闭
extern void DT_Set_Is_Print_New_PC(int isPrintPC);
 
// 18.设置单个测试例运行时间，如果设置的运行次数没到而运行时间到了，提前结束
extern void DT_Set_Running_Time_Second(int second);
 
// 18.1为了与测试例最低运行要求配合，当产生新样本时，
// 1.运行次数重新计算，2，运行时间重新计算，3，前两者都重新计算,0和默认都是不做事情
extern void DT_Set_Reset_On_New_Corpus(int value);
 
// 19.打开样本进化过程对循环语句的支持，因为会一定程序影响测试效率，默认开启
extern void DT_Enable_Support_Loop(int isEnable);
 
// 20.使能每运行一次测试用例检测一次内存泄露，默认开启
// 检测到内存泄漏后，复现过程中，可以打开isDebug用来打印内存分配释放过程方便定位
extern void DT_Enable_Leak_Check(int isEnable, int isDebug);
 
// 21.被测代码在运行时，调用下边函数，会使当时的变异样本被保存下来，用于以后变异
extern void DT_Set_HasNewFeature(int weight);
 
extern int DT_Get_HasNewFeature(int *weight);
 
 
// 22.设置是否删除不匹配的样本文件
// 如果设置删除1，则程序删除样本后可以继续执行，
// 如果设置不删除0，则程序遇到不匹配的样本直接退出
// 默认不删除0
extern void DT_Set_IsDeleteMismatchCopusFile(int isDelete);
 
// 24.支持多线程时，用来线程退出前删除工具分配的内存，避免内存泄露
extern void DT_Clear_ThreadMemory(void);
 
// 25.设置一个权重值影响多少变异次数,默认1000
extern void DT_Set_MutatorCountFromWeight(int value);
 
// 26,27.28用于同步变异
extern int DT_Get_RandomSeed(void);
 
extern void DT_Set_RandomSeed(int value);
 
extern int DT_Get_RandomVlaue(void);
 
// 29.设置是否注册asan报告回调，默认使能，在报告没去重的场景需要关闭
extern void DT_Enable_AsanReport(int isEnable);
 
// 30.存放二进制样本路径，仅一个测试用例有效，过后清除
extern void DT_SetBinCorpusDirIn(char *dir, int maxLen);
 
extern void DT_SetBinCorpusDirOut(char *dir);
 
// 31.使能fork模式，DT循环在子进程中运行,默认关闭
// isEnable  1 使能fork模式，每个用例等待  2，使能fork，每个用例不等待，相当于并行
extern void DT_SetEnableFork(int isEnable);
 
// 32.设置外部字典文件路径，单测试用例有效
extern void DT_SetDictionaryPath(char *path);
 
// 33.设置样本产生不同产生方式的权重
// 1,代码块覆盖(30),
// 2,跳转覆盖(3),
// 3,循环覆盖(20),
// 4,分支覆盖(1)，
// 以上设置为0则这种方式不对样本产生有影响
// 5,设置写道样本文件里的样本权重，默认20(不是所有内存中的样本都被写道样本文件里)
// 当被测代码有定时器的时候，2要调整为0
extern void DT_SetPathModeWeight(int mode, int weight);
 
/* 由使用者自定义，实现外部变异算法
 size_t DT_CustomMutator(uint8_t* data, size_t size, size_t max_size, unsigned int seed);
 size_t DT_CustomCrossOver(const uint8_t *data, size_t size1,const uint8_t *data2, size_t size2, uint8_t *out,size_t
 max_size, unsigned int Seed);
*/
 
// 34.注册回调函数，每发现crash时工具调用
extern void DT_AddCrashCallBack(void (*fun)(void));
 
// 35.注册回调函数，每个测试用例运行完成工具调用
extern void DT_AddTestCaseCallBack(void (*fun)(void));
 
// 36.设置监控OutOfMemory，MallocLimitMb默认2048M,一次分配内存最大值,RssLimitMb默认2048M,物理内存rss使用最大值
extern void DT_SetCheckOutOfMemory(int mallocLimitMb, int rssLimitMb);
 
// 默认没有开启检测内存耗尽功能
extern void DT_SetStopOutOfMemory(void);
 
// 37.获取上个测试用例是否运行成功
extern int DT_GetIsPass(void);
 
// 38.设置获取不到trace-pc信息，用例直接退出循环，默认为0，不退出，设置为1，退出
extern void DT_SetNoTracePCStop(int stop);
 
// 39.设置仅运行样本，然后退出，一般仅用于覆盖率获取
extern void DT_SetOnlyRunCorpus(int isOnlyRunCorpus);
 
// 40.使能crash时重新拉起进程执行 默认(0, 0)
// isEnable  0 不重启进程  1 重启进程
// reStartNum  进程重启次数
extern void DT_SetEnableExec(int isEnable, int reStartNum);
 
// 41.设置用例文件路径和函数名 打印在种子文件前两行
// file_path: <fileNameWithPath>
// test_func_name: <testFuncName>
// name		=xxx
// type		=xxx
// in_type		=xxx
// ...
// !!!!!!!!!!!!!!!!!!!!!!!!!!!above is corpus 1, time is xxxx_xxxx_xx_xx_xx
extern void DT_SetTcPathAndFunc(char *fileNameWithPath, char *testFuncName);
 
// 可以使用这个函数传递命令行，改变运行参数，配合DT_FUZZ_START1使用
extern int DT_Cmd(int argc, char *argv[]);
 
 
/* ***********************************************************************
 
1.封装测试例循环的声明
 
DT_FUZZ_START(0, 30000000,"test_example" ,0)
{
    printf("\r%d" ,fuzzSeed + fuzzi);
 
    sec_s32 number= *(sec_s32 *)DT_SetGetS32(&g_Element[0],0x123456);
    char *string = DT_SetGetString(&g_Element[1], 10, 10000,"zhangpeng" );
    char *buf = DT_SetGetBlob(&g_Element[2], 6, 20000,"12345" );
    int  buf_len=DT_GET_MutatedValueLen(&g_Element[2]);
 
    fun_example(number,string, buf , buf_len);
}
DT_FUZZ_END()
 
说明:
被测函数的上下文须放在DT_FUZZ_START和DT_FUZZ_END循环体里
DT_FUZZ_START(seed, count, testCaseName, isReproduce)
seed            为随机变异使用的seed值，一般设置为0，工具会随机个seed出来
count           为测试次数
testCaseName    为样本文件路径和名称
isReproduce     是否为复现模式,0为测试模式；1为复现模式，只根据指定样本运行1次
 
注:
1.DT_SetGet函数调用不要在if语句里，这样会导致调用跳变
2.DT_SetGet函数初始值如果是内存指针，要在循环外部初始化，
避免循环下次执行时，作用域失效导致初始值指针变化
3.测试用例要与样本逻辑对应，如果测试用例有修改，请删除样本再进行测试
4.变异出的内存，工具会自己释放，如果被测程序要释放内存，
请将变异后的内存copy出来再调用被测函数
5.工具生成的样本是一种测试资源，再次运行测试用例工具能够读取样本继续测试，
如果测试用例逻辑没有修改，多次测试请不要删除样本
 
*********************************************************************** */
#ifdef SUPPORT_M_THREAD
extern __thread int fuzzSeed;
extern __thread int fuzzi;
extern __thread int fuzzStart;
extern __thread int fuzzEnd;
extern __thread SElementInit *g_Element;
extern __thread int fuzzPid;
extern __thread int fuzzStatus;
extern __thread int g_isHasNewFeature;
 
#else
extern int fuzzSeed;
extern int fuzzi;
extern int fuzzStart;
extern int fuzzEnd;
extern SElementInit *g_Element;
extern int fuzzPid;
extern int fuzzStatus;
extern int g_isHasNewFeature;
 
#endif
 
extern void DTStart(int seed, int count, char *testcaseName, int isReproduce);
 
extern void DTForStart(void);
 
extern void DTForEnd(void);
 
extern void DTEnd(void);
 
extern int DTStop(void);
 
extern int DTFork(void);
 
extern void DTWait(char *tempTestCaseName);
 
extern void DTExit(int no);
 
extern int RunningTimeIsOver(void);
 
#define DT_FUZZ_START(seed, count, testCaseName, isReproduce)   \
    {                                                           \
        fuzzPid = DTFork();                                     \
        char *tempTestCaseName = testCaseName;                  \
        if (fuzzPid == 0) {                                     \
            DTStart(seed, count, testCaseName, isReproduce);    \
            for (fuzzi = fuzzStart; fuzzi < fuzzEnd; fuzzi++) { \
                DTForStart();
 
// 没有清理内存的地方，好尴尬，一定要清理内存的话，就自己调用fork等函数吧
#define DT_FUZZ_END()                      \
    DTForEnd();                            \
    if (RunningTimeIsOver() || DTStop()) { \
        break;                             \
    }                                      \
    }                                      \
    DTEnd();                               \
    DTExit(DT_GetIsPass());                \
    }                                      \
    else                                   \
    {                                      \
        DTWait(tempTestCaseName);          \
    }                                      \
    }
 
 
#define DT_FUZZ_END_EX()                   \
    DTForEnd();                            \
    if (RunningTimeIsOver() || DTStop()) { \
        break;                             \
    }                                      \
    }                                      \
    DTEnd();                               \
    }                                      \
    else                                   \
    {                                      \
        DTWait(tempTestCaseName);          \
    }                                      \
    }
 
 
/* ***********************************************************************
 
0.内置命令行处理
 
*********************************************************************** */
 
extern int g_dtRunCount;          // 运行次数,默认1亿次
extern int g_dtRunTime;           // 运行时间，默认12小时
extern int g_dtIIsReproduce;      // 是否是复现模式
extern int g_dtIsLeakCheck;       // 是否检测内存泄露
extern char *g_dtReportPath;      // 报告路径
extern char *g_dtCorpusPath;      // 样本路径
extern char *g_dtCrashCorpusName; // 复现模式时，传递复现样本名称
extern char g_dtTestCaseName[];
 
 
#define DT_FUZZ_START_CMD(testCaseName)                              \
    char *testcaseName = testCaseName;                               \
    if (g_dtCrashCorpusName)                                         \
        testcaseName = g_dtCrashCorpusName;                          \
    g_dtTestCaseName[0] = 0;                                         \
    sprintf(g_dtTestCaseName, "%s%s", g_dtCorpusPath, testcaseName); \
    DT_FUZZ_START(0, g_dtRunCount, g_dtTestCaseName, g_dtIIsReproduce)
 
 
// 对外部提供的接口，第一个单词大写，然后用下划线与后边隔开，其他每个单词首字母大写
 
/* *****************************************
 
2.接口一DT_SetGet系列接口
 
注:
 
1)
获取返回内存得长度举例如下
char *buf = DT_SetGetBlob(&g_Element[2], 6, 20000,"12345" );
int  buf_len=DT_GET_MutatedValueLen(&g_Element[2]);
 
2)
内存类参数获取，初始样本的内存和初始长度必须匹配
正确
char *string = DT_SetGetString(&g_Element[1], 10, 10000,"zhangpeng" );
错误
char *string = DT_SetGetString(&g_Element[1], 64, 10000,"zhangpeng" );
 
3)
第一个参数为固定为 &g_Element[]，下标从0开始，依次递增，不允许跳变
 
4)
同一个下标只能有一个setget函数调用，下列写法错误
if (a == 1)
{
    DT_SetGetS32(&g_Element[0])
}
else
{
    DT_SetGetString(&g_Element[0])
}
 
***************************************** */
 
// 数据变异接口扩展，增加默认参数 char* desc
// desc 自定义描述如被测代码入参变量名，用于快速定位种子文件变量在用例中的位置
#define DT_SetGetS64(init, initValue, ...) DTSetGetS64(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetS32(init, initValue, ...) DTSetGetS32(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetS16(init, initValue, ...) DTSetGetS16(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetS8(init, initValue, ...) DTSetGetS8(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetU64(init, initValue, ...) DTSetGetU64(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetU32(init, initValue, ...) DTSetGetU32(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetU16(init, initValue, ...) DTSetGetU16(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetU8(init, initValue, ...) DTSetGetU8(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetFloat(init, initValue, ...) DTSetGetFloat(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetDouble(init, initValue, ...) DTSetGetDouble(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetFLoat16(init, initValue, ...) DTSetGetFLoat16(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetFLoat32(init, initValue, ...) DTSetGetFLoat32(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetFLoat64(init, initValue, ...) DTSetGetFLoat64(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetDouble64(init, initValue, ...) DTSetGetDouble64(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberEnum(init, initValue, eunmTable, eunmCount, ...) \
    DTSetGetNumberEnum(init, initValue, eunmTable, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberEnum_EX(init, initValue, eunmTable, eunmCount, ...) \
    DTSetGetNumberEnum_EX(init, initValue, eunmTable, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberEnumS64(init, initValue, eunmTable, eunmCount, ...) \
    DTSetGetNumberEnumS64(init, initValue, eunmTable, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberEnumS64_EX(init, initValue, eunmTable, eunmCount, ...) \
    DTSetGetNumberEnumS64_EX(init, initValue, eunmTable, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberEnum_FUZZ(init, initValue, eunmTable, eunmCount, ...) \
    DTSetGetNumberEnum_FUZZ(init, initValue, eunmTable, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberRange(init, initValue, min, max, ...) \
    DTSetGetNumberRange(init, initValue, min, max, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberRange_EX(init, initValue, min, max, ...) \
    DTSetGetNumberRange_EX(init, initValue, min, max, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberRangeS64(init, initValue, min, max, ...) \
    DTSetGetNumberRangeS64(init, initValue, min, max, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberRangeS64_EX(init, initValue, min, max, ...) \
    DTSetGetNumberRangeS64_EX(init, initValue, min, max, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberRangeDouble(init, initValue, min, max, ...) \
    DTSetGetNumberRangeDouble(init, initValue, min, max, (NULL, ##__VA_ARGS__))
#define DT_SetGetNumberRangeFloat(init, initValue, min, max, ...) \
    DTSetGetNumberRangeFloat(init, initValue, min, max, (NULL, ##__VA_ARGS__))
#define DT_SetGetString(init, length, maxLength, initValue, ...) \
    DTSetGetString(init, length, maxLength, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetStringNum(init, length, maxLength, initValue, ...) \
    DTSetGetStringNum(init, length, maxLength, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetStringEnum(init, length, maxLength, initValue, eunmTableS, eunmCount, ...) \
    DTSetGetStringEnum(init, length, maxLength, initValue, eunmTableS, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetStringEnum_EX(init, length, maxLength, initValue, eunmTableS, eunmCount, ...) \
    DTSetGetStringEnum_EX(init, length, maxLength, initValue, eunmTableS, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetStringEnum_FUZZ(init, length, maxLength, initValue, eunmTableS, eunmCount, ...) \
    DTSetGetStringEnum_FUZZ(init, length, maxLength, initValue, eunmTableS, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetFixString(init, length, maxLength, initValue, ...) \
    DTSetGetFixString(init, length, maxLength, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetBlob(init, length, maxLength, initValue, ...) \
    DTSetGetBlob(init, length, maxLength, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetBlobEnum(init, length, maxLength, initValue, eunmTableB, eunmTableL, eunmCount, ...) \
    DTSetGetBlobEnum(init, length, maxLength, initValue, eunmTableB, eunmTableL, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetBlobEnum_EX(init, length, maxLength, initValue, eunmTableB, eunmTableL, eunmCount, ...) \
    DTSetGetBlobEnum_EX(init, length, maxLength, initValue, eunmTableB, eunmTableL, eunmCount, (NULL, ##__VA_ARGS__))
#define DT_SetGetBlobEnum_FUZZ(init, length, maxLength, initValue, eunmTableB, eunmTableL, eunmCount, ...) \
    DTSetGetBlobEnum_FUZZ(init, length, maxLength, initValue, eunmTableB, eunmTableL, eunmCount,         \
        (NULL, ##__VA_ARGS__))
#define DT_SetGetFixBlob(init, length, maxLength, initValue, ...) \
    DTSetGetFixBlob(init, length, maxLength, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetIpv4(init, initValue, ...) DTSetGetIpv4(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetIpv6(init, initValue, ...) DTSetGetIpv6(init, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetMac(Sinit, initValue, ...) DTSetGetMac(Sinit, initValue, (NULL, ##__VA_ARGS__))
#define DT_SetGetSelf(init, length, maxLength, initValue, arg, ...) \
    DTSetGetSelf(init, length, maxLength, initValue, arg, (NULL, ##__VA_ARGS__))
 
 
// init 元素结构体指针,使用&g_Element[n],每个测试例n从0开始，必须连续
// initValue 初始值或指向初始值的指针
// 返回值,指向变异值的指针,
// 相应得类型只返回相应大小得内存，比如s32返回得指针为4个字节
// 获取返回值举例:sec_s32 number= *(sec_s32 *)DT_SetGetS32
char *DTSetGetS64(SElementInit *init, sec_s64 initValue, char *desc);
 
char *DTSetGetS32(SElementInit *init, sec_s32 initValue, char *desc);
 
char *DTSetGetS16(SElementInit *init, sec_s16 initValue, char *desc);
 
char *DTSetGetS8(SElementInit *init, sec_s8 initValue, char *desc);
 
char *DTSetGetU64(SElementInit *init, sec_u64 initValue, char *desc);
 
char *DTSetGetU32(SElementInit *init, sec_u32 initValue, char *desc);
 
char *DTSetGetU16(SElementInit *init, sec_u16 initValue, char *desc);
 
char *DTSetGetU8(SElementInit *init, sec_u8 initValue, char *desc);
 
char *DTSetGetFloat(SElementInit *init, float initValue, char *desc);
 
char *DTSetGetDouble(SElementInit *init, double initValue, char *desc);
 
char *DTSetGetFLoat16(SElementInit *init, char *initValue, char *desc);
 
char *DTSetGetFLoat32(SElementInit *init, char *initValue, char *desc);
 
char *DTSetGetFLoat64(SElementInit *init, char *initValue, char *desc);
 
char *DTSetGetDouble64(SElementInit *init, char *initValue, char *desc);
 
/* 变异后的每个值都在枚举表里，不会变异出其他的
 例如
 ! initValue               = 0x1234
 ! int eunmTable[]    ={2,4,6,1234,0x1234,0x897654,0x456789}
 ! eunmCount           =7
 ! desc                ="aa"
 返回值,指向变异值的指针,4个字节
 获取返回值举例:sec_s32 number= *(sec_s32 *)DT_SetGetNumberEnum
*/
char *DTSetGetNumberEnum(SElementInit *init, sec_s32 initValue, int *eunmTable, int eunmCount, char *desc);