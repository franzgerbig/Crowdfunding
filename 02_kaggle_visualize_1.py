# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 14:14:16 2023

@authors: Hendrik Bosse, Franz Gerbig
"""


# ---------------------------------------------------------------

# load dataset created in exploration
filename=r'data\kaggle\Kaggle_deduplicated.csv'
df_final=pd.read_csv(filename)

### Visualization

## BEFORE preprocessing
cols=['backers_count','usd_pledged','goal_usd','launched_year','duration']
# (no?) correlations betwwen explanatory variables
sns.heatmap(df_final[cols].corr(),cmap='winter',annot=True)
plt.title('Correlation of variables');

sns.pairplot(df_final[cols],diag_kind='kde')
plt.title('Distribution of variables');

# relation between backers and converted pledged amount
plt.scatter(df_final['backers_count'],df_final['usd_pledged'])
plt.plot((df_final["backers_count"].min(),df_final["backers_count"].max()),(df_final["usd_pledged"].min(),                                                             df_final["usd_pledged"].max()),"red")
plt.xlabel('number of backers')
plt.ylabel('pledged amount')
plt.title('backers and pledged amount');


# Development of funding goals by project (top 3 of main) category and status over time
select=df_final["main_category"].value_counts()[:3].index.tolist()
g=sns.catplot(x="launched_year",y="backers_count",hue="status",row="cat_parent_name",\
            row_order=select,col="status",kind="bar",errorbar=('ci',False),height=4,data=df_final);
g.set_axis_labels("Year of project launch","Number of backers")
plt.tick_params(bottom='off',labelbottom='on')
# plt.xticks(rotation=30) # does it only for the very last subplot
g.set_xticklabels(rotation=30,ha="right")
# add margin to top of plot grid (to havc enough space for grid title)
g.fig.subplots_adjust(top=.93)
# modify titles of subplots
g.set_titles("{row_name} ({col_name})")
# add general grid title
g.fig.suptitle("Development of backers' quantity by project (top 3 of main) category and status over time");


## Statistics
# ANOVA for numerical explanatory variables
num_list=['backers_count',
       'usd_pledged','goal_usd',
       'launched_year','duration']
import statsmodels.api
for item in num_list:
    print('###',item)
    result=statsmodels.formula.api.ols(f'{item} ~ status',data=df_final).fit()
    display(statsmodels.api.stats.anova_lm(result))
    
# chi2 for categorical explanatory variables   
cat_list=['currency','country','main_category',"launched_day"]
import scipy.stats as sc
cat_relevant=[]
cat_drop=[]
for item in cat_list:
    chi2=sc.chi2_contingency(pd.crosstab(df_final['status'],df_final[item]))[0]
    pvalue=sc.chi2_contingency(pd.crosstab(df_final['status'],df_final[item]))[1]
    print(item,':',"chi2:",chi2,"p-value:",pvalue)
    if pvalue <=0.05:
        cat_relevant.append(item)
    else:
        cat_drop.append(item)
