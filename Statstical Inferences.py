#import the libraries we will use in the code
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
from scipy.stats import norm
import scipy.stats
import statistics
from scipy import stats
# define the path of the dataset
url='https://drive.google.com/file/d/1x333Xl9yNXGFb2v47ZviG2gkjijRrsAj/view?usp=share_link'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
# read the dataset to make operations on the data and filter it
df = pd.read_csv(url)
# filter the data to include only people who have a degree and received the training
treatment= df.loc[df["treat"]==1]
degree_holder= treatment.loc[treatment["nodegree"]==0]
# Break the dataset into two groups, black group and white group respectively
black_people_group=degree_holder.loc[degree_holder["black"]==1]
white_people_group= degree_holder.loc[degree_holder["black"]==0]
# eliminate people who did not receive income in 1978
white_people_group_income= white_people_group.loc[white_people_group["re78"]>0]
black_people_group_income= black_people_group.loc[ black_people_group["re78"]>0]
# remove the outlier from the dataset
Black_people_group_income_nooutliers= black_people_group_income.loc[ black_people_group_income["re78"] < 30000]
# calculate the sample number of both groups
white_people_group_income=white_people_group_income["re78"]
Black_people_group_income_nooutliers=Black_people_group_income_nooutliers["re78"]
n_white=len(white_people_group_income)
n_black=len(Black_people_group_income_nooutliers)
print("the sample number of the black people group is" +"" + str(n_black))
print("the sample number of the white people group is" + ""+ str(n_white))
# calculate the mean, median, mode, range and standard deviation of the white group
white_mean=np.mean(white_people_group_income)
std_white= np.std(white_people_group_income)
median_white=np.median(white_people_group_income)
range_white= max(white_people_group_income)- min(white_people_group_income)
print("the mean of white people income"+str(white_mean))
print("the Standard deviation of white people income"+str(std_white))
print("the median of white people income"+str(median_white))
print("the range of white people income"+str(range_white))
# calculate the mean, median, mode, range and standard deviation of the black group
black_mean=np.mean(Black_people_group_income_nooutliers)
std_black= np.std(Black_people_group_income_nooutliers)
median_black=np.median(Black_people_group_income_nooutliers)
range_black= max(Black_people_group_income_nooutliers)- min(Black_people_group_income_nooutliers)
print("the mean of black people income"+str(black_mean))
print("the Standard deviation of black people income"+str(std_black))
print("the median of black people income"+str(median_black))
print("the range of black people income"+str(range_black))
# plot histograms for both groups to present the frequency of the income

white_people_group_income.hist(color="b",bins=10,edgecolor="black")
plt.xlabel("degree-holders White People income after taking the training")
plt.ylabel("Frequency")
# xlim was used to unify the x-axis of both groups
# unifying x-axis ease the observation of differences between both graphs
plt.xlim([0,26000])
plt.show()

Black_people_group_income_nooutliers.hist(color="orange",bins=10,edgecolor="black")
plt.xlabel("degree-holders Black People income after taking the training")
plt.ylabel("Frequency")
plt.xlim([0,26000])

plt.show()

# plotting the QQ plots of the income of both groups to measure the normality of the distribution
stats.probplot(white_people_group_income, plot=plt)
plt.ylabel="number of degree-holders white People income after taking the training"
plt.show()
stats.probplot(Black_people_group_income_nooutliers, plot=plt)
plt.ylabel="number of degree-holders black People income after taking the training"
plt.show()


# calculating the a one_sided 95% confidence interval of the differences between the two means of the income of black and white people
means_difference= white_mean - black_mean # calculate the difference between the two mean
# this will be the point estimate
standard_error= np.sqrt((std_white**2)/n_white + (std_black**2)/n_black)# calculate the standard error
df= min((n_black-1), (n_white-1))# calculate the degree of freedom
t_value= stats.t.ppf(1-(1-0.95)/2,df) #setting the level of confidence
# because we are doing one-sided test, we will not multiply it by 2
print("the confidence interval is" + "[" +str(means_difference+t_value*-int(standard_error))+"," +str(means_difference+t_value*standard_error)+"]" )


means_difference= white_mean - black_mean # calculate the difference between the two mean
standard_error= np.sqrt((std_white**2)/n_white + (std_black**2)/n_black)# calculate the standard error
t_score= np.abs(means_difference)/standard_error # calculate the t-score
p_value= 1- (stats.t.cdf(t_score,df)) # calculate the p-value
print("the p_value is "  + str(p_value))
print("the t_value is " + str(t_score))
# Calculate the pooled Standadrd deviation, cohens d and hedges d
SD_pooled = np.sqrt((std_black ** 2 + std_white ** 2) / 2)
cohen_d = ((means_difference) / SD_pooled)
Hedge_g = cohen_d * (1 - (3 / (4 * (n_white + n_black) - 9)))
print(" hedges'g is " + str(Hedge_g))

# calculate the power of the test
t_score=stats.t.ppf(0.9,16)# determine the T-score which represents the 90% significance level
#  the T-score will be used to mark the rejection regions
mean_1= 0 # the mean of the null hypothesis
sd_1=1429.94 # the standard deviation of the null hypothesis
x_axis= np.arange(mean_1+sd_1*-4,mean_1+sd_1*4,0.001)# setting the x-axis to extend by 4 standard deviations from the mean
plt.plot(x_axis, norm.pdf(x_axis, mean_1, sd_1)) # plotting the curve
mean_2= -2884.50911020893 # the mean of the alternative hypothesis
sd_2=1429.9477100360712 # the standard deviation of the alternative hypothesis
# the same as the null because they are the same graph but shifted
plt.axvline(sd_1*t_score, color='r', linestyle='--',linewidth=1) # drawing a line that will present the rejection region
x_axis_2= np.arange(mean_2+sd_2*-4,mean_2+sd_2*4,0.001) # setting the x-axis to extend by 4 standard deviations from the mean
px=np.arange(-8000,sd_1*t_score,0.001) # defining the shaded region
iq=stats.norm(mean_2,sd_2)
plt.fill_between(px,iq.pdf(px),color = "orange") # shading the area which represents the power of the test
plt.plot(x_axis_2, norm.pdf(x_axis_2, mean_2, sd_2))
plt.xlabel("Mean of the difference between the income of Black people and White people")
plt.show()

power= ((sd_1*t_score) - mean_2)/sd_1
print("The power of the test is" + str(stats.t.cdf(power,44+17-2)))  # calculating the power of the test