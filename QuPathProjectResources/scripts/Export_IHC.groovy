import qupath.lib.scripting.QP
import qupath.lib.io.PathIO
import qupath.lib.objects.PathObjects
import qupath.lib.gui.dialogs.Dialogs
import java.io.File

def ANNOTATION_COLUMNS = [
    'Image',
    'Object ID',
    'Object type',
    'Name',
    'Classification',
    'Parent',
    'ROI',
    'Centroid X µm',
    'Centroid Y µm',
    'Num Detections',
    'Num CardiomyocyteNuclei',
    'Num Connexin',
    'Area µm^2',
    'Perimeter µm'    
]

def DETECTION_COLUMNS = [
   'Image',
   'Object ID',
   'Object type',
   'Name',
   'Classification',
   'Parent',
   'ROI',
   'Centroid X µm',
   'Centroid Y µm',
   'Nucleus: Area',
   'Nucleus: Perimeter',
   'Nucleus: Circularity',
   'Nucleus: Max caliper',
   'Nucleus: Min caliper',
   'Nucleus: Eccentricity'
]

def project = getProject()
if (project == null) {
    Dialogs.showErrorMessage("Export Error","No project is open")
    return
}

def projectDir = project.getPath() != null ? project.getPath().getParent().toFile() : null
def outputDir = Dialogs.promptForDirectory("Select output directory for CSV exports: ", projectDir)
if (outputDir == null) {
    print("Export cancelled - No output directory")
    return
}
if (!outputDir.canWrite()) {
    Dialogs.showErrorMessage("Directory Error","Cannot write to selected directory.")
    return
}

def exportedCount = 0
def errorCount = 0

def projectEntries = project.getImageList()
for (entry in projectEntries) {
    try {
        print("Processing: " + entry.getImageName())
        
        def imageName = entry.getImageName()
        def baseName = imageName.lastIndexOf('.') > 0 ?
            imageName.substring(0, imageName.lastIndexOf('.')) : imageName
        
        def imageData = entry.readImageData()
        // Removed the problematic line: QP.setImageData(imageData)
        
        def annotationFile = new File(outputDir, baseName + "-Annotation.csv")
        def annotations = imageData.getHierarchy().getAnnotationObjects()
        
        if (annotations.size() > 0) {
            saveAnnotations(annotationFile, ANNOTATION_COLUMNS, annotations, imageData)
            print("Exported Annotations")
        } else {
            print("No Annotations found")
            createEmptyCSV(annotationFile, ANNOTATION_COLUMNS)
        }
        
        def detectionFile = new File(outputDir, baseName + "-Detection.csv")
        def detections = imageData.getHierarchy().getDetectionObjects()
        
        if (detections.size() > 0) {
            saveDetections(detectionFile, DETECTION_COLUMNS, detections, imageData)
            print("Exported Detections")
        } else {
            print("No Detections found")
            createEmptyCSV(detectionFile, DETECTION_COLUMNS)
        }
        
        exportedCount++
        print("Exported: " + imageName)
        
    } catch (Exception e) {
        errorCount++
        print("Error processing: " + entry.getImageName() + ": " + e.getMessage())
        e.printStackTrace()
    }
}

print("EXPORTS COMPLETE for: " + exportedCount + " images")
print("Errors: " + errorCount + " images")

def createEmptyCSV(file, columns) {
    file.getParentFile().mkdirs()
    file.withWriter('UTF-8') { writer -> 
        writer.writeLine(columns.join(','))
    }
}

def saveAnnotations(file, columns, annotations, imageData) {
    file.getParentFile().mkdirs()
    file.withWriter('UTF-8') { writer ->
        writer.writeLine(columns.join(','))
        
        for (annotation in annotations) {
            def values = []
             for (column in columns) {
                 def value = getObjectValue(annotation, column, imageData)
                 values.add(value != null ? value.toString(): '')
             }
             writer.writeLine(values.join(','))
        }
    }
}

def saveDetections(file, columns, detections, imageData) { 
    file.getParentFile().mkdirs()
    file.withWriter('UTF-8') { writer ->
        writer.writeLine(columns.join(','))
        
        for (detection in detections) {
            def values = []
             for (column in columns) {
                 def value = getObjectValue(detection, column, imageData)
                 values.add(value != null ? value.toString(): '')
             }
             writer.writeLine(values.join(','))
        }
    }
}

def getObjectValue(pathObject, columnName, imageData) { 
    try {
        if (columnName == 'Image') {
            return imageData.getServer().getMetadata().getName()
        } else if (columnName == 'Object ID') {
            return pathObject.getID()?.toString() ?: ''
        } else if (columnName == 'Object type') {
            return pathObject.getClass().getSimpleName()
        } else if (columnName == 'Name') {
            return pathObject.getDisplayedName() ?: pathObject.getPathClass()?.toString() ?: 'Unclassified'
        } else if (columnName == 'Classification') {
            return pathObject.getPathClass()?.toString() ?: 'Unclassified'
        } else if (columnName == 'Parent') {
            def parent = pathObject.getParent()
            return parent != null ? parent.getDisplayedName() ?: parent.getPathClass()?.toString() : 'Root'
        } else if (columnName == 'ROI') {
            return 'Polygon'
        } else if (columnName == 'Centroid X µm') {
            return pathObject.getROI()?.getCentroidX() ? String.format("%.3f", pathObject.getROI().getCentroidX()) : ''
        } else if (columnName == 'Centroid Y µm') {
            return pathObject.getROI()?.getCentroidY() ? String.format("%.3f", pathObject.getROI().getCentroidY()) : ''
        } else if (columnName == 'Num Detections') {
            return pathObject.getChildObjects().size()
        } else if (columnName == 'Num CardiomyocyteNuclei') {
            return pathObject.getChildObjects().findAll { it.getPathClass()?.toString() == 'CardiomyocyteNuclei' }.size()
        } else if (columnName == 'Num Connexin') {
            return pathObject.getChildObjects().findAll { it.getPathClass()?.toString() == 'Connexin' }.size()
        } else if (columnName == 'Area µm^2') {
            return pathObject.getROI()?.getArea() ? String.format("%.3f", pathObject.getROI().getArea()) : ''
        } else if (columnName == 'Perimeter µm') {
            return pathObject.getROI()?.getLength() ? String.format("%.3f", pathObject.getROI().getLength()) : ''
        } else if (columnName == 'Circularity') {
            return pathObject.getMeasurementList().getMeasurementValue("Circularity") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Circularity")) : ''
        } else if (columnName == 'Max diameter µm') {
            return pathObject.getMeasurementList().getMeasurementValue("Max diameter µm") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Max diameter µm")) : ''
        } else if (columnName == 'Min diameter µm') {
            return pathObject.getMeasurementList().getMeasurementValue("Min diameter µm") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Min diameter µm")) : ''
        } else if (columnName == 'Nucleus: Area' || columnName == 'Nucleus: Area µm^2') {
            return pathObject.getMeasurementList().getMeasurementValue("Nucleus: Area") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Nucleus: Area")) : ''
        } else if (columnName == 'Nucleus: Perimeter' || columnName == 'Nucleus: Perimeter µm') {
            return pathObject.getMeasurementList().getMeasurementValue("Nucleus: Perimeter") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Nucleus: Perimeter")) : ''
        } else if (columnName == 'Nucleus: Circularity') {
            return pathObject.getMeasurementList().getMeasurementValue("Nucleus: Circularity") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Nucleus: Circularity")) : ''
        } else if (columnName == 'Nucleus: Max caliper') {
            return pathObject.getMeasurementList().getMeasurementValue("Nucleus: Max caliper") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Nucleus: Max caliper")) : ''
        } else if (columnName == 'Nucleus: Min caliper') {
            return pathObject.getMeasurementList().getMeasurementValue("Nucleus: Min caliper") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Nucleus: Min caliper")) : ''
        } else if (columnName == 'Nucleus: Eccentricity') {
            return pathObject.getMeasurementList().getMeasurementValue("Nucleus: Eccentricity") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Nucleus: Eccentricity")) : ''
        } else if (columnName == 'Cell: Area µm^2') {
            return pathObject.getMeasurementList().getMeasurementValue("Cell: Area µm^2") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Cell: Area µm^2")) : ''
        } else if (columnName == 'Cell: Perimeter µm') {
            return pathObject.getMeasurementList().getMeasurementValue("Cell: Perimeter µm") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Cell: Perimeter µm")) : ''
        } else if (columnName == 'Cytoplasm: Area µm^2') {
            return pathObject.getMeasurementList().getMeasurementValue("Cytoplasm: Area µm^2") != null ? String.format("%.3f", pathObject.getMeasurementList().getMeasurementValue("Cytoplasm: Area µm^2")) : ''
        } else {
            // Try to get as a custom measurement
            def measurement = pathObject.getMeasurementList().getMeasurementValue(columnName)
            return measurement != null ? String.format("%.3f", measurement) : ''
        }
    } catch (Exception e) {
        print("Warning: Could not get value for column: " + columnName + ": " + e.getMessage())
        return ''
    }
}