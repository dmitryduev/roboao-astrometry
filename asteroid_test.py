import matplotlib
matplotlib.use('Qt5Agg')
import numpy as num, astropy.io.fits as pyf, pylab as pyl
from trippy import psf, pill, psfStarChooser
from trippy import scamp,MCMCfit
import scipy as sci
from os import path
import os
from stsci import numdisplay
from astropy.visualization import interval


def trimCatalog(cat):
    good=[]
    for i in range(len(cat['XWIN_IMAGE'])):
        try:
            a=int(cat['XWIN_IMAGE'][i])
            b=int(cat['YWIN_IMAGE'][i])
            m=num.max(data[b-4:b+5,a-4:a+5])
        except: pass
        dist = num.sort(((cat['XWIN_IMAGE']-cat['XWIN_IMAGE'][i])**2+(cat['YWIN_IMAGE']-cat['YWIN_IMAGE'][i])**2)**0.5)
        d=dist[1]
        if cat['FLAGS'][i]==0 and d>30 and m<70000:
            good.append(i)
    good=num.array(good)
    outcat={}
    for i in cat:
        outcat[i]=cat[i][good]
    return outcat

inputFile = 'Polonskaya.fits'
if not path.isfile(inputFile):
    os.system('wget -O Polonskaya.fits http://www.canfar.phys.uvic.ca/vospace/nodes/fraserw/Polonskaya.fits?view=data')

with pyf.open(inputFile) as han:
    data=han[0].data
    header=han[0].header
    EXPTIME=header['EXPTIME']

scamp.makeParFiles.writeSex('example.sex',
                            minArea=3.,
                            threshold=5.,
                            zpt=27.8,
                            aperture=20.,
                            min_radius=2.0,
                            catalogType='FITS_LDAC',
                            saturate=55000)
scamp.makeParFiles.writeConv()
scamp.makeParFiles.writeParam(numAps=1)  # numAps is thenumber of apertures that you want to use. Here we use 1

scamp.runSex('example.sex', inputFile, options={'CATALOG_NAME': 'example.cat'}, verbose=False)
catalog = trimCatalog(scamp.getCatalog('example.cat', paramFile='def.param'))

dist = ((catalog['XWIN_IMAGE']-811)**2+(catalog['YWIN_IMAGE']-4005)**2)**0.5
args = num.argsort(dist)
xt = catalog['XWIN_IMAGE'][args][0]
yt = catalog['YWIN_IMAGE'][args][0]

rate=18.4588 # "/hr
angle=31.11+1.1 # degrees counter clockwise from horizontal, right

starChooser = psfStarChooser.starChooser(data,
                                         catalog['XWIN_IMAGE'], catalog['YWIN_IMAGE'],
                                         catalog['FLUX_AUTO'], catalog['FLUXERR_AUTO'])
(goodFits, goodMeds, goodSTDs) = starChooser(30, 200, noVisualSelection=False, autoTrim=True)
print goodFits
print goodMeds

goodPSF = psf.modelPSF(num.arange(61), num.arange(61), alpha=goodMeds[2], beta=goodMeds[3], repFact=10)
fwhm = goodPSF.FWHM()  # this is the pure moffat FWHM.
# Can also get this value by passing option fromMoffatProfile=True
goodPSF.genLookupTable(data, goodFits[:, 4], goodFits[:, 5], verbose=False)
fwhm = goodPSF.FWHM()  # this is the FWHM with look uptable included
fwhm = goodPSF.FWHM(fromMoffatProfile=True)  # this is the pure moffat FWHM.

print "Full width at half maximum {:5.3f} (in pix).".format(fwhm)

(z1, z2) = numdisplay.zscale.zscale(goodPSF.lookupTable)
normer = interval.ManualInterval(z1, z2)
pyl.imshow(normer(goodPSF.lookupTable))
pyl.show()

goodPSF.line(rate,angle,EXPTIME/3600.,pixScale=0.185,useLookupTable=True)

goodPSF.computeRoundAperCorrFromPSF(psf.extent(0.8*fwhm,4*fwhm,10),display=False,
                                                          displayAperture=False,
                                                          useLookupTable=True)
roundAperCorr=goodPSF.roundAperCorr(1.4*fwhm)

goodPSF.computeLineAperCorrFromTSF(psf.extent(0.1*fwhm,4*fwhm,10),
                                   l=(EXPTIME/3600.)*rate/0.185,a=angle,display=False,displayAperture=False)
lineAperCorr=goodPSF.lineAperCorr(1.4*fwhm)
print lineAperCorr, roundAperCorr

goodPSF.psfStore('psf.fits')

#goodPSF=psf.modelPSF(restore='psf.fits')
#goodPSF.line(new_rate,new_angle,EXPTIME/3600.,pixScale=0.185,useLookupTable=True)

#initiate the pillPhot object
phot=pill.pillPhot(data,repFact=10)
#get photometry, assume ZPT=26.0
#enableBGselection=True allows you to zoom in on a good background region in the aperture display window
#trimBGhighPix is a sigma cut to get rid of the cosmic rays. They get marked as blue in the display window
#background is selected inside the box and outside the skyRadius value
#mode is th background mode selection. Options are median, mean, histMode (JJ's jjkmode technique), fraserMode (ask me about it), gaussFit, and "smart". Smart does a gaussian fit first, and if the gaussian fit value is discrepant compared to the expectation from the background std, it resorts to the fraserMode. "smart" seems quite robust to nearby bright sources

#examples of round sources
phot(goodFits[0][4], goodFits[0][5],radius=3.09*1.1,l=0.0,a=0.0,
             skyRadius=4*3.09,width=6*3.09,
              zpt=26.0,exptime=EXPTIME,enableBGSelection=True,display=True,
              backupMode="fraserMode",trimBGHighPix=3.)

#example of a trailed source
phot(xt,yt,radius=fwhm*1.4,l=(EXPTIME/3600.)*rate/0.185,a=angle,
             skyRadius=4*fwhm,width=6*fwhm,
              zpt=26.0,exptime=EXPTIME,enableBGSelection=True,display=True,
              backupMode="smart",trimBGHighPix=3.)

phot.SNR(verbose=True)

#get those values
print phot.magnitude
print phot.dmagnitude
print phot.sourceFlux
print phot.snr
print phot.bg

phot.computeRoundAperCorrFromSource(goodFits[0,4],goodFits[0,5],num.linspace(1*fwhm,4*fwhm,10),
                                    skyRadius=5*fwhm, width=6*fwhm,displayAperture=False,display=True)
print phot.roundAperCorr(1.4*fwhm)

Data=data[int(yt)-200:int(yt)+200,int(xt)-200:int(xt)+200]-phot.bg

fitter=MCMCfit.MCMCfitter(goodPSF,Data)
fitter.fitWithModelPSF(200+xt-int(xt)-1,200+yt-int(yt)-1, m_in=1000.,
                       fitWidth=10,
                       nWalkers=20, nBurn=20, nStep=20,
                       bg=phot.bg, useLinePSF=True, verbose=False,useErrorMap=False)

(fitPars,fitRange)=fitter.fitResults(0.67)
print fitPars
print fitRange

modelImage=goodPSF.plant(fitPars[0],fitPars[1],fitPars[2],Data,addNoise=False,useLinePSF=True,returnModel=True)
pyl.imshow(modelImage)
pyl.show()

removed=goodPSF.remove(fitPars[0],fitPars[1],fitPars[2],Data,useLinePSF=True)

(z1,z2)=numdisplay.zscale.zscale(removed)
normer=interval.ManualInterval(z1,z2)

pyl.imshow(normer(Data))
pyl.show()

pyl.imshow(normer(removed))
pyl.show()

