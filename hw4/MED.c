#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define min(a, b, c) (a < b ? (a < c ? a : c) : (b < c ? b : c))

struct MED
{
    char str[100];
    int cost;
} typedef MED;

int min_edit_distance(char *, char *);
char *optimal_path(char *, char *);

int main()
{
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    char str1[] = "intensation";
    FILE *fp = fopen("wordlist.txt", "r");

    if (fp == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }

    MED min_5[5];
    for (int i = 0; i < 5; i++)
    {
        min_5[i].cost = 1000;
    }

    while (!feof(fp))
    {
        char str2[100];
        fscanf(fp, "%s", str2);
        int cost = min_edit_distance(str1, str2);
        if (cost < min_5[4].cost)
        {
            min_5[4].cost = cost;
            strcpy(min_5[4].str, str2);
        }
        for (int i = 0; i < 4; i++)
        {
            if (min_5[i].cost > min_5[i + 1].cost)
            {
                MED temp;
                temp.cost = min_5[i].cost;
                min_5[i].cost = min_5[i + 1].cost;
                min_5[i + 1].cost = temp.cost;
                strcpy(temp.str, min_5[i].str);
                strcpy(min_5[i].str, min_5[i + 1].str);
                strcpy(min_5[i + 1].str, temp.str);
            }
        }
    }

    min_edit_distance("intention", "execution");
    printf("The word for spelling correction: %s\n", str1);
    for (int i = 0; i < 5; i++)
    {
        printf("Candidate %d: %-13s %d OptPath: %s\n", i + 1, min_5[i].str, min_5[i].cost, optimal_path(str1, min_5[i].str));
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %.2f microseconds(us)\n", cpu_time_used * 1000000);
    fclose(fp);
}

int min_edit_distance(char *str1, char *str2)
{
    int n = strlen(str1);
    int m = strlen(str2);
    int dp[n + 1][m + 1];

    for (int i = 0; i <= n; i++)
        dp[i][0] = i;
    for (int j = 0; j <= m; j++)
        dp[0][j] = j;

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            int cost = (str1[i - 1] == str2[j - 1]) ? 0 : 2;
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
        }
    }

    // for(int i = 0; i <= n; i++)
    // {
    //     for(int j = 0; j <= m; j++)
    //     {
    //         printf("%4d", dp[i][j]);
    //     }
    //     printf("\n");
    // }

    return dp[n][m];
}

char *optimal_path(char *str1, char *str2)
{
    int n = strlen(str1);
    int m = strlen(str2);
    int dp[n + 1][m + 1];
    static char path[1000];
    int path_idx = 0;

    // 初始化 DP 表格
    for (int i = 0; i <= n; i++)
        dp[i][0] = i;
    for (int j = 0; j <= m; j++)
        dp[0][j] = j;

    // 填充 DP 表格
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            int cost = (str1[i - 1] == str2[j - 1]) ? 0 : 2;
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
        }
    }

    // 回溯尋找最佳路徑
    int i = n, j = m;
    path[0] = '\0';

    while (i > 0 || j > 0)
    {
        if (i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] && str1[i - 1] == str2[j - 1])
        {
            path_idx += sprintf(path + path_idx, "%c", str1[i - 1]); // 相同字符印出原本字元
            i--;
            j--;
        }
        else if (i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 2)
        {
            path_idx += sprintf(path + path_idx, "*"); // 替換為*號
            i--;
            j--;
        }
        else if (i > 0 && dp[i][j] == dp[i - 1][j] + 1)
        {
            path_idx += sprintf(path + path_idx, "-"); // 刪除為-號
            i--;
        }
        else
        {
            path_idx += sprintf(path + path_idx, "+"); // 插入為+號
            j--;
        }
    }

    // 反轉路徑字串
    for (int i = 0, j = strlen(path) - 1; i < j; i++, j--)
    {
        char temp = path[i];
        path[i] = path[j];
        path[j] = temp;
    }

    return path;
}