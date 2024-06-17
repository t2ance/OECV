# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea


class DE_best_1_L(ea.SoeaAlgorithm):
    '''
    Adapted from ea.soea_SEGA_templet, only the run method is changed

    '''

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing,
                         trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1:
            raise RuntimeError()
        self.name = 'DE/best/1/L'
        self.selFunc = 'ecs'
        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)
            self.recOper = ea.Xovexp(XOVR=0.5, Half_N=True)
        else:
            raise RuntimeError()

    def deversify(self, prophetPop, population, retain_size: bool):
        sizes = population.sizes
        population = prophetPop + population
        if retain_size:
            population = population[: sizes]
        return population

    def call_aimFunc(self, pop):
        pop.Phen = pop.decoding()
        if self.problem is None:
            raise RuntimeError('error: problem has not been initialized.')
        self.problem.evaluation(pop)
        self.evalsNum = self.evalsNum + pop.sizes if self.evalsNum is not None else pop.sizes
        if not isinstance(pop.ObjV, np.ndarray) or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != len(pop.CV) or \
                pop.ObjV.shape[1] != self.problem.M:
            raise RuntimeError('error: ObjV is illegal.')
        if pop.CV is not None:
            if not isinstance(pop.CV, np.ndarray) or pop.CV.ndim != 2 or pop.CV.shape[0] != len(pop.CV):
                raise RuntimeError('error: CV is illegal.')

    def run(self, prophetPop=None):
        if prophetPop is None:
            prophetPop = ea.Population(Encoding='RI', NIND=0)
        prophetPop.sizes = prophetPop.Chrom.shape[0]
        population = self.population
        NIND = population.sizes
        self.initialization()
        population.initChrom(population.sizes)
        self.call_aimFunc(prophetPop)
        if self.MAXGEN == 0:
            population = prophetPop + population
            self.call_aimFunc(population)
            population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)
            population = population[ea.selecting('otos', population.FitnV, NIND)]
            self.check(population)
            self.stat(population)
        else:
            for _ in range(self.MAXGEN):
                population = prophetPop + population
                self.call_aimFunc(population)
                population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)
                r0 = ea.selecting(self.selFunc, population.FitnV, population.sizes)
                experimentPop = ea.Population(population.Encoding, population.Field, population.sizes)
                experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field, [r0])
                experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))
                self.call_aimFunc(experimentPop)
                tempPop = population + experimentPop
                tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)
                population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]
                self.check(population)
                self.stat(population)
        return self.finishing(population)
