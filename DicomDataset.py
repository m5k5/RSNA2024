from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import glob
import re

class OrientationType(Enum):
    Sagittal= "Sagittal"
    Axial="Axial"
    Frontal="Frontal"
    Unknown="Unknown"

class Direction(Enum):
    X=0
    Y=1
    Z=2


class Slice:
    def __init__(self, data, series, patId, position, orientation, sliceThickness, pixelSpacing):
        self.series = series 
        self.patId = patId 
        self.position = position 
        self.orientation = orientation 
        self.sliceThickness = sliceThickness
        self.pixelSpacing = pixelSpacing
        self.data = data

    def plot(self, hlines=None):
        plt.figure()
        plt.imshow(self.data)
        if hlines:
            plt.hlines(hlines, int(self.data.shape[0]*0.1), int(self.data.shape[0]*0.9), "red")

    def getWorldPosition(self,x,y):
        shiftX = x*self.pixelSpacing[0]
        shiftY = y*self.pixelSpacing[1]
        row_cosines = np.array(self.orientation[:3])  
        col_cosines = np.array(self.orientation[3:]) 
        world_position = (
            np.array(self.position) +
            shiftX * row_cosines +
            shiftY * col_cosines
        )
        return np.array(world_position)



class Scan:
    def __init__(self, folder, orientationType: OrientationType):
        self.orientationType = orientationType
        eps=10**(-12)
        allFilesInDir = glob.glob(f"{folder}/*.dcm")
        allFilesInDir = sorted(allFilesInDir, key=lambda x:int(re.findall(r"\d+\.dcm", x)[0].replace(".dcm", "")))
        self.slices=[]
        for f in allFilesInDir:
            dicom = pydicom.read_file(f)
            data = pydicom.pixel_data_handlers.util.apply_modality_lut(dicom.pixel_array, dicom)
            data = pydicom.pixel_data_handlers.util.apply_windowing(data, dicom)
            data = data - np.min(data)
            data = data * 255.0/(np.max(data)+eps)
            seriesDescr = dicom.SeriesDescription
            patId = dicom.PatientID
            imPos = dicom.ImagePositionPatient
            imOr = dicom.ImageOrientationPatient
            slTh = dicom.SliceThickness
            pxSp = dicom.PixelSpacing
            slice = Slice(data.astype(np.uint8), seriesDescr, patId, imPos, imOr, slTh, pxSp)
            self.slices.append(slice)


class PatientData:
    def __init__(self, scanMapping:list[tuple[OrientationType, str]]):
        self.scans=[]
        for orientationType, folder in scanMapping:
            self.scans.append(Scan(folder, orientationType))

    def getAxialScans(self):
        axialScans = []
        for s in self.scans:
            if s.orientationType==OrientationType.Axial:
                axialScans.append(s)
        return axialScans
    
    def getSagittalScans(self):
        sagScans = []
        for s in self.scans:
            if s.orientationType==OrientationType.Sagittal:
                sagScans.append(s)
        return sagScans
    
    def getClosestSliceInScan(self, scan:Scan, position):
        positions = [s.position for s in scan.slices]
        distances = [np.linalg.norm(position-p) for p in positions]
        minIdx = np.argmin(distances)
        return scan.slices[minIdx]
    
    def getClosestSliceInScanDirection(self, scan:Scan, position, direction:Direction, threshold=1.0):
        positions = [s.position for s in scan.slices]
        distances = np.abs([position[direction.value]-p[direction.value] for p in positions])
        minIdx = np.argmin(distances)
        if distances[minIdx]>threshold:
            raise Exception(f"No image found closer than {threshold}")
        return scan.slices[minIdx]
    
    def getSlicesInRangeDirection(self, scan:Scan, minPos, maxPos, direction:Direction):
        positions = [s.position for s in scan.slices]
        foundSlices=[]
        if minPos[direction.value] >= maxPos[direction.value]:
            raise ValueError(f"minPos ({minPos}) has to be smaller than maxPos{maxPos} in the specified direction {direction}")
        for i,p in enumerate(positions):
            if p[direction.value]>minPos[direction.value] and p[direction.value]<maxPos[direction.value]:
                foundSlices.append(scan.slices[i])
        return foundSlices
