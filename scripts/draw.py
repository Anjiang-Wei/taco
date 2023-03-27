import matplotlib.pyplot as plt

x = [i for i in range(1, 8)]
# y = [5000 * i for i in range(1, 6)]
labels = [str(2**i) + " (" + str(2**i * 4) + ")" for i in range(0, len(x))]
plt.xticks(x, labels, fontsize=20)
# plt.yticks(y, fontsize=20)

cpp = dsl = tune = None

def cannon():
    global cpp, dsl, tune
    cpp = [27117.9661,9292.142102,13372.16883,0,12376.63895,0,8899.986094]
    dsl = [27117.9661,9765.371609,14828.359592,0,12931.82461,0,10914.01774]
    tune = [7637.040573,4326.265013,6168.003084,3216.056572,6646.54689,2987.807476,6257.619164]

def pumma():
    global cpp, dsl, tune
    cpp = [27210.20408,16336.72637,16023.83575,0,12648.14229,0,9151.326232]
    dsl = [27026.35135,16138.63482,15391.82299,0,13448.11935,0,9732.329684]
    tune = [7622.486899,5187.154062,5818.109091,0,5716.802144,0,6570.484061]

def summa():
    global cpp, dsl, tune
    cpp = [26844.96644,15756.5223,15678.39295,0,12576.06603,0,8448.818482]
    dsl = [27256.55877,15205.89114,14492.57246,0,12673.18812,0,8673.822593]
    tune = [7607.988588,4796.770409,6092.840823,0,6011.609994,0,5418.43119]

def solomonik():
    global cpp, dsl, tune
    cpp = [28267.84452,14010.13376,11790.56743,7116.391261,10657.71857,5743.173047,8850.753699]
    dsl = [28217.98942,14053.64349,12038.97667,6581.316467,10435.28453,5410.165823,10717.54166]
    tune = [28118.80492,14311.42696,16251.70137,8217.383699,11506.58037,5102.128871,0]

def johnson():
    global cpp, dsl, tune
    cpp = [27680.96886,15154.96719,10796.08637,7473.67994,9400.6463]
    dsl = [27728.94281,15144.8233,10869.42935,6842.021776,9396.505652]
    tune = [29519.5572,12812.21179,18680.44367,10110.17248,0]

def cosma():
    global cpp, dsl, tune
    cpp = [21857.37705,15175.29579,13366.58312,8527.064636,9421.404388]
    dsl = [22252.57302,15054.13574,13417.02306,8468.026204,9407.555492]
    tune = [20024.53066,12848.58945,15332.82223,0,0]

def circuit():
    global cpp, dsl
    wires = 20000 * 40 * 50
    ctime = [1.1964,1.21,1.2144,1.3848,1.2278,1.3448,1.3484]
    dtime = [1.1994,1.2046,1.209,1.3782,1.2272,1.3404,1.3488]
    cpp = [wires / i / 1e7 for i in ctime]
    dsl = [wires / i / 1e7 for i in dtime]
    y = [i * 0.5 for i in range(0, 8)]
    plt.ylim([0, 3.5])
    plt.yticks(y, fontsize=20)
    plt.xlabel("Nodes (GPUs)", fontsize=20)
    plt.ylabel("Thoughput Per Node ($10^{7}$ wires/s)", fontsize=20)

def stencil():
    global cpp, dsl
    cells = 4 * 50 * 15000 * 15000
    ctime = [0.5578,0.5776,0.5952,0.6052,0.6112,0.6204,0.6312]
    dtime = [0.558,0.5774,0.5954,0.6052,0.6126,0.6202,0.6306]
    cpp = [cells / i / 1e10 for i in ctime]
    dsl = [cells / i / 1e10 for i in dtime]
    y = [i * 2 for i in range(0, 5)]
    plt.ylim([0, 10])
    plt.yticks(y, fontsize=20)
    plt.xlabel("Nodes (GPUs)", fontsize=20)
    plt.ylabel("Thoughput Per Node ($10^{10}$ cells/s)", fontsize=20)

def pennant():
    global cpp, dsl
    zones = 30 * 320 * 92160
    ctime = [1.2746,1.3586,1.3804,1.4068,1.4818,1.5388,1.5562]
    dtime = [1.2758,1.3892,1.401,1.4342,1.4766,1.5366,1.5528]
    cpp = [zones / i / 1e8 for i in ctime]
    dsl = [zones / i / 1e8 for i in dtime]
    y = [i for i in range(0, 8)]
    plt.ylim([0, 8])
    plt.yticks(y, fontsize=20)
    plt.xlabel("Nodes (GPUs)", fontsize=20)
    plt.ylabel("Thoughput Per Node ($10^{8}$ zones/s)", fontsize=20)

cannon()
# pumma()
# summa()
# solomonik()
# johnson()
# cosma()

# circuit()
# stencil()
# pennant()

length = len(cpp)
plt.plot(x[:length], cpp, "--x", label="C++", linewidth=3, markersize=16)
plt.plot(x[:length], dsl, "--o", label="Centurian", linewidth=3, markersize=16)
if tune != None:
    plt.plot(x[:length], tune, "--^", label="Tuned", linewidth=3, markersize=16)
    y = [5000 * i for i in range(1, 6)]
    plt.yticks(y, fontsize=20)
    plt.xlabel("Nodes (GPUs)", fontsize=20)
    plt.ylabel("GFLOP/s Per Node", fontsize=20)
# plt.plot(x, full_db, "--+", label="full - db")

plt.legend(fontsize=20)
plt.show()
