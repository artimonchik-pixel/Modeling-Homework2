'''
ЭМ волна в виде модулированного гассова импульса распространяется
в одну сторону (TFSF boundary).
Источник находится в диэлектрике.
'''

import numpy as np

import tools


class GaussianModPlaneWave:
    ''' Класс с уравнением плоской волны для модулированного гауссова сигнала в дискретном виде
    d - определяет задержку сигнала.
    w - определяет ширину сигнала.
    Nl - количество ячеек на длину волны.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''

    def __init__(self, d, w, Nl, Sc=1.0, eps=1.0, mu=1.0):
        self.d = d
        self.w = w
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return (np.sin(2 * np.pi / self.Nl * (q * self.Sc - m * np.sqrt(self.eps * self.mu))) *
                np.exp(-(((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - self.d) / self.w) ** 2))


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Число Куранта
    Sc = 1.0
    #Скорость света
    c=3e8
    # Параметры сигнала
    eps_1=9
    fmin=0.5e9
    fmax=3.5e9
    f0=(fmax+fmin)/2
    DeltaF=fmax-fmin
    A_0=100
    A_max=100
    wg=2 * np.sqrt(np.log(A_max)) / (np.pi * DeltaF)
    dg=wg * np.sqrt(np.log(A_0))
    #Шаг по пространству в метрах
    dx=c/(fmax*20)/np.sqrt(eps_1)
    Nl=c/(f0*dx)
    #шаг во времени в секундах
    dt=(Sc*dx)/c
    #Дискретизация параметров сигнала
    wg=wg/dt
    dg=dg/dt
    # Время расчета в секундах
    maxTime_sec = 3.3e-8
    maxTime=int(np.ceil(maxTime_sec/dt))
    # Размер области моделирования в метрах
    maxSize_m = 4
    maxSize=int(np.ceil(maxSize_m/dx))
    # Положение источника в метрах
    sourcePos_m = 1
    sourcePos=int(np.ceil(sourcePos_m/dx))
    # Датчики для регистрации поля в метрах
    probesPos = [3]
    probes = [tools.Probe(int(np.ceil(pos/dx)), maxTime) for pos in probesPos]
    
    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)
    eps[:] = eps_1

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    source = GaussianModPlaneWave(
        dg, wg, Nl, Sc, eps[sourcePos], mu[sourcePos])

    # Sc' для правой границы
    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])
    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)
    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = np.zeros(3)
    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = np.zeros(3)
    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = np.zeros(3)
    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = np.zeros(3)
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1
    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,dx,dt)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy[:] = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)
        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getE(0, q)
        ##PMC LEFT
        Hy[0]=0
        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * W0 / eps[1:-1]
        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))
        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]
        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)
        if q % 10 == 0:
            display.updateData(display_field, q)

    display.stop()
    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1,dx,dt,maxTime)
    #отображение спектра
    tools.Spectrum(f0,DeltaF,wg,dg)
