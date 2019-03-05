import csv
import math

import numpy as np

from GaussianND import Gaussian

# df_barrel_blue = pd.read_csv("barrel_blue.csv")
# df_barrel_blue.columns = ['H', 'S', 'V']
#
# df_nonbarrel_blue = pd.read_csv("non-barrel_blue.csv")
# df_nonbarrel_blue.columns = ['H', 'S', 'V']
#
# df_brown = pd.read_csv("brown.csv")
# df_brown.columns = ['H', 'S', 'V']
#
# df_sky_black = pd.read_csv("sky_black.csv")
# df_sky_black.columns = ['H', 'S', 'V']
#
# df_sky_white = pd.read_csv("sky_white.csv")
# df_sky_white.columns = ['H', 'S', 'V']
# #print(df.columns)
# #print(df.head(n = 3))
#
# barrel_blue_H = df_barrel_blue.H.values
# barrel_blue_S = df_barrel_blue.S.values
# barrel_blue_V = df_barrel_blue.V.values
#
# nonbarrel_blue_H = df_nonbarrel_blue.H.values
# nonbarrel_blue_S = df_nonbarrel_blue.S.values
# nonbarrel_blue_V = df_nonbarrel_blue.V.values
#
# brown_H = df_brown.H.values
# brown_S = df_brown.S.values
# brown_V = df_brown.V.values
#
# df_sky_black_H = df_sky_black.H.values
# df_sky_black_S = df_sky_black.S.values
# df_sky_black_V = df_sky_black.V.values
#
# sky_white_H = df_sky_white.H.values
# sky_white_S = df_sky_white.S.values
# sky_white_V = df_sky_white.V.values
#
# #H = H.to_int(index = False)
# #print(barrel_blue_H)
#
# data = np.concatenate((nonbarrel_blue_H, sky_white_H))
# # print(type(data))
# # print(type(nonbarrel_blue_H))
# #data = np.concatenate(data, nonbarrel_blue_H)
# #print(data.shape)
# # sns.distplot(H, bins = 20, kde = False)
# # sns.distplot(S, bins = 20, kde = False)
# # sns.distplot(V, bins = 20, kde = False)
# # plt.show()
#
# # best fit using single Gaussian model:
# best_single = Gaussian(np.mean(data), np.std(data))
# print('Best single Gaussian: mu = {:.2}, sigma = {:.2}'.format(best_single.mu, best_single.sigma))
#
# # # fit a single gaussian curve to the data
# x = np.linspace(0, 250, 500)
# g_single = stats.norm(best_single.mu, best_single.sigma).pdf(x)
# sns.distplot(data, bins = 20, kde = False, norm_hist=True)
# plt.plot(x, g_single, label = 'single Gaussian')
# plt.legend()
# plt.show()

with open('gaussian_other_colors.csv') as input:
    reader = csv.reader(input, delimiter=',', quotechar='|')
    other_class_mu = [[float(ele)] for ele in next(reader)]
    other_class_cov = [[float(row[0]), float(row[1]), float(row[2])] for row in reader]

other_class_mu = np.array(other_class_mu)
other_class_cov = np.array(other_class_cov)

data = []
with open('other_colors.csv') as input:
    reader = csv.reader(input, delimiter=',', quotechar='|')
    data = [[float(row[0]), float(row[1]), float(row[2])] for row in reader]
data = np.asarray(data)


class Gaussian_Mixture:
    ''''
    try two Gaussians and their EM estimation.
    '''

    # def __init__(self, data, mu_min=100, mu_max=100, sigma_min=.1, sigma_max=1, mix=.5):
    #     self.data = data
    #     # init with multiple gaussians
    #     self.one = Gaussian(uniform(mu_min, mu_max),
    #                         uniform(sigma_min, sigma_max))
    #     self.two = Gaussian(uniform(mu_min, mu_max),
    #                         uniform(sigma_min, sigma_max))
    #
    #     # as well as how much to mix them
    #     self.mix = mix

    def __init__(self, data, mu=other_class_mu, cov=other_class_cov, wp=np.empty(10), mix=[0.1] * 10):
        self.data = data
        # init with multiple gaussians
        self.one = Gaussian(mu, cov)
        self.two = Gaussian(mu, cov)
        self.three = Gaussian(mu, cov)
        self.four = Gaussian(mu, cov)
        self.five = Gaussian(mu, cov)
        self.six = Gaussian(mu, cov)
        self.seven = Gaussian(mu, cov)
        self.eight = Gaussian(mu, cov)
        self.nine = Gaussian(mu, cov)
        self.ten = Gaussian(mu, cov)

        # as well as how much to mix them
        self.mix = mix
        self.wp = wp

        self.loglike = 0

    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        print("Start computing Estep")
        # compute weights
        self.loglike = 0
        for datum in self.data:
            datum = np.transpose([datum])

            # unnormalized weights
            # print(datum)
            # print('one:', self.one)
            # print('two:', self.two)
            # print('one.pdf:', self.one.pdf(datum))
            # print('mix:', self.mix)
            # wp1 = self.one.pdf(datum) * self.mix[0]
            # wp2 = self.two.pdf(datum) * (1 - self.mix)

            # print(self.one.pdf(datum))
            # print(datum.shape)
            # print(self.one.mu.shape)

            # print(self.one.pdf(datum))
            # print(self.two.pdf(datum))
            # self.wp[1] = 1

            self.wp[0] = np.asscalar(self.one.pdf(datum)) * self.mix[0]
            # print('wp1 calculated')
            self.wp[1] = np.asscalar(self.two.pdf(datum)) * self.mix[1]
            self.wp[2] = np.asscalar(self.three.pdf(datum)) * self.mix[2]
            self.wp[3] = np.asscalar(self.four.pdf(datum)) * self.mix[3]
            self.wp[4] = np.asscalar(self.five.pdf(datum)) * self.mix[4]
            self.wp[5] = np.asscalar(self.six.pdf(datum)) * self.mix[5]
            self.wp[6] = np.asscalar(self.seven.pdf(datum)) * self.mix[6]
            self.wp[7] = np.asscalar(self.eight.pdf(datum)) * self.mix[7]
            self.wp[8] = np.asscalar(self.nine.pdf(datum)) * self.mix[8]
            self.wp[9] = np.asscalar(self.ten.pdf(datum)) * self.mix[9]

            # print('wp1:', wp1)
            # print('wp2:', wp2)
            # compute denominator
            den = sum(self.wp)
            # print('den:', den)
            # normalize
            self.wp /= den
            # add into loglike
            self.loglike += math.log(sum(self.wp))
            # yield weight tuple
            yield (self.wp)

    def Mstep(self, weights):
        "Perform an M(aximization)-step"
        print("Start computing Mstep")
        # compute denominators
        (one, two, three, four, five, six, seven, eight, nine, ten) = zip(*weights)
        one_den = sum(one)
        two_den = sum(two)
        three_den = sum(three)
        four_den = sum(four)
        five_den = sum(five)
        six_den = sum(six)
        seven_den = sum(seven)
        eight_den = sum(eight)
        nine_den = sum(nine)
        ten_den = sum(ten)

        # compute new means
        self.one.mu = sum(w * d / one_den for (w, d) in zip(one, data))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(two, data))
        self.three.mu = sum(w * d / three_den for (w, d) in zip(three, data))
        self.four.mu = sum(w * d / four_den for (w, d) in zip(four, data))
        self.five.mu = sum(w * d / five_den for (w, d) in zip(five, data))
        self.six.mu = sum(w * d / six_den for (w, d) in zip(six, data))
        self.seven.mu = sum(w * d / seven_den for (w, d) in zip(seven, data))
        self.eight.mu = sum(w * d / eight_den for (w, d) in zip(eight, data))
        self.nine.mu = sum(w * d / nine_den for (w, d) in zip(nine, data))
        self.ten.mu = sum(w * d / ten_den for (w, d) in zip(ten, data))

        # compute new sigmas
        self.one.sigma = math.sqrt(sum(w * ((d - self.one.mu) ** 2)
                                       for (w, d) in zip(one, data)) / one_den)

        self.two.sigma = math.sqrt(sum(w * ((d - self.two.mu) ** 2)
                                       for (w, d) in zip(two, data)) / two_den)

        self.three.sigma = math.sqrt(sum(w * ((d - self.three.mu) ** 2)
                                         for (w, d) in zip(three, data)) / three_den)

        self.four.sigma = math.sqrt(sum(w * ((d - self.four.mu) ** 2)
                                        for (w, d) in zip(four, data)) / four_den)

        self.five.sigma = math.sqrt(sum(w * ((d - self.five.mu) ** 2)
                                        for (w, d) in zip(five, data)) / five_den)

        self.six.sigma = math.sqrt(sum(w * ((d - self.six.mu) ** 2)
                                       for (w, d) in zip(six, data)) / six_den)

        self.seven.sigma = math.sqrt(sum(w * ((d - self.seven.mu) ** 2)
                                         for (w, d) in zip(seven, data)) / seven_den)

        self.eight.sigma = math.sqrt(sum(w * ((d - self.eight.mu) ** 2)
                                         for (w, d) in zip(eight, data)) / eight_den)

        self.nine.sigma = math.sqrt(sum(w * ((d - self.nine.mu) ** 2)
                                        for (w, d) in zip(nine, data)) / nine_den)

        self.ten.sigma = math.sqrt(sum(w * ((d - self.ten.mu) ** 2)
                                       for (w, d) in zip(ten, data)) / ten_den)
        # compute new mix
        self.mix[0] = one_den / len(data)
        self.mix[1] = two_den / len(data)
        self.mix[2] = three_den / len(data)
        self.mix[3] = four_den / len(data)
        self.mix[4] = five_den / len(data)
        self.mix[5] = six_den / len(data)
        self.mix[6] = seven_den / len(data)
        self.mix[7] = eight_den / len(data)
        self.mix[8] = nine_den / len(data)
        self.mix[9] = ten_den / len(data)

    def iterate(self, N=10, verbose=False):
        "Perform N iterations, then compute log-likelihood"
        self.Mstep(self.Estep())

    def pdf(self, x):
        return (self.mix[0]) * self.one.pdf(x) + (self.mix[1]) * self.two.pdf(x) + \
               (self.mix[2]) * self.three.pdf(x) + (self.mix[3]) * self.four.pdf(x) + (self.mix[4]) * self.five.pdf(x) + \
               (self.mix[5]) * self.six.pdf(x) + (self.mix[6]) * self.seven.pdf(x) + (self.mix[7]) * self.eight.pdf(x) + \
               (self.mix[8]) * self.nine.pdf(x) + (self.mix[9]) * self.ten.pdf(x)

    # def __repr__(self):
    #     return 'GaussianMixture({0}, {1}, mix={2.03})'.format(self.one,
    #                                                           self.two,
    #                                                           self.mix)
    # def __str__(self):
    #     return 'Mixture: {0}, {1}, mix={2:.03}'.format(self.one, self.two, self.mix)


# check the first couple of iterations:
# Check out the fitting process
n_iterations = 5
best_mix = None
best_loglike = float('-inf')
g_mix = Gaussian_Mixture(data)
# counter = 0
for i in range(n_iterations):
    # counter += 1
    # print(counter)
    g_mix.iterate(verbose=True)
    # mix.Mstep(mix.Estep())
    if g_mix.loglike > best_loglike:
        best_loglike = g_mix.loglike
        best_mix = g_mix
    print('1st try:', 'epoch:', i, 'mix:', best_mix, 'log:', best_loglike)

# Find best Mixture Gaussian model
n_iterations = 20
n_random_restarts = 150
best_mix = None
best_loglike = float('-inf')
print('Computing best model with random restarts...\n')
counter = 0
for i in range(n_random_restarts):
    g_mix = Gaussian_Mixture(data)
    for _ in range(n_iterations):
        g_mix.iterate()
        counter += 1
        if g_mix.loglike > best_loglike:
            best_loglike = g_mix.loglike
            best_mix = g_mix
    print('2nd try:', 'epoch:', i, 'mix:', best_mix, 'log:', best_loglike)

print(type(best_mix))
print(best_loglike)
print(best_mix)
print(counter)

# # show mixture:
# sns.distplot(data, bins = 20, kde = False, norm_hist=True)
# g_both = [best_mix.pdf(e) for e in x]
# plt.plot(x, g_both, label = 'Gaussian Mixture Two')
# plt.legend()
# plt.show()
