<?php
// At the top of the file, change these lines
ini_set('display_errors', 1);
error_reporting(E_ALL);

// And add this to explicitly log errors
ini_set('log_errors', 1);
ini_set('error_log', '/var/log/apache2/error.log');

header('Content-Type: application/json');

// Define base directory
$baseDir = __DIR__;  // This will be /var/www/html/hablas

// Function to return error response
function returnError($message) {
    echo json_encode(['error' => $message]);
    exit;
}

// Check for required PHP extensions
if (!extension_loaded('zip')) {
    returnError('ZIP extension not loaded');
}

if (!extension_loaded('xml')) {
    returnError('XML extension not loaded');
}

// Check if directories exist and are readable
if (!is_dir($baseDir . '/epub')) {
    returnError('EPUB directory does not exist: ' . $baseDir . '/epub');
}

if (!is_dir($baseDir . '/covers')) {
    returnError('Covers directory does not exist: ' . $baseDir . '/covers');
}

if (!is_readable($baseDir . '/epub')) {
    returnError('EPUB directory is not readable: ' . $baseDir . '/epub');
}

function extractEpubCover($epubPath, $coverPath, $baseDir) {
    try {
        $zip = new ZipArchive();
        if ($zip->open($epubPath) === TRUE) {
            // First, find the container.xml
            $container = $zip->getFromName('META-INF/container.xml');
            if ($container) {
                try {
                    $xml = new SimpleXMLElement($container);
                    $rootfile = $xml->rootfiles->rootfile['full-path'];
                    
                    // Read the OPF file
                    $opf = $zip->getFromName($rootfile);
                    if ($opf) {
                        $opfXml = new SimpleXMLElement($opf);
                        $opfDir = dirname($rootfile);
                        
                        // Look for cover image in metadata
                        $coverIds = [];
                        foreach ($opfXml->metadata->meta as $meta) {
                            if ((string)$meta['name'] === 'cover') {
                                $coverIds[] = (string)$meta['content'];
                            }
                        }
                        
                        // Find the cover image path
                        foreach ($opfXml->manifest->item as $item) {
                            $id = (string)$item['id'];
                            if (in_array($id, $coverIds) || 
                                strpos(strtolower($id), 'cover') !== false ||
                                strpos(strtolower((string)$item['href']), 'cover') !== false) {
                                
                                $coverImagePath = $item['href'];
                                if ($opfDir !== '.') {
                                    $coverImagePath = $opfDir . '/' . $coverImagePath;
                                }
                                
                                // Extract the cover image
                                $imageData = $zip->getFromName($coverImagePath);
                                if ($imageData) {
                                    $extension = pathinfo($coverImagePath, PATHINFO_EXTENSION);
                                    
                                    // Create language directory in covers if it doesn't exist
                                    $coverDir = dirname($coverPath);
                                    error_log("Attempting to create directory: " . $coverDir);
                                    
                                    if (!is_dir($coverDir)) {
                                        if (!mkdir($coverDir, 0777, true)) {
                                            error_log("Failed to create directory: " . $coverDir);
                                            error_log("Error: " . error_get_last()['message']);
                                        }
                                    }
                                    
                                    $tempFile = basename($epubPath, '.epub') . '.' . $extension;
                                    $fullCoverPath = $coverPath . '/' . $tempFile;
                                    
                                    // Make sure parent directory exists
                                    $parentDir = dirname($fullCoverPath);
                                    if (!is_dir($parentDir)) {
                                        if (!mkdir($parentDir, 0777, true)) {
                                            error_log("Failed to create parent directory: " . $parentDir);
                                            error_log("Error: " . error_get_last()['message']);
                                            return false;
                                        }
                                    }
                                    
                                    error_log("Attempting to save file to: " . $fullCoverPath);
                                    if (!file_put_contents($fullCoverPath, $imageData)) {
                                        error_log("Failed to save file: " . $fullCoverPath);
                                        error_log("Error: " . error_get_last()['message']);
                                        return false;
                                    }
                                    
                                    error_log("Successfully saved cover to: " . $fullCoverPath);
                                    return str_replace($baseDir . '/', '', $fullCoverPath);
                                }
                            }
                        }
                    }
                } catch (Exception $e) {
                    error_log("XML Error: " . $e->getMessage());
                    return false;
                }
            }
            $zip->close();
        }
    } catch (Exception $e) {
        error_log("Error processing EPUB: " . $e->getMessage());
        return false;
    }
    return false;
}

try {
    $books = [];
    
    // Get all language directories in epub folder
    $languageDirs = glob($baseDir . '/epub/*', GLOB_ONLYDIR);
    
    foreach ($languageDirs as $langDir) {
        $language = basename($langDir);
        $epubs = glob($langDir . '/*.epub');
        
        foreach ($epubs as $epub) {
            $filename = basename($epub);
            // Create corresponding cover path
            $coverDir = $baseDir . '/covers/' . $language;
            
            $coverPath = extractEpubCover($epub, $coverDir, $baseDir);
            
            $books[] = [
                'filename' => $filename,
                'language' => $language,
                'path' => str_replace($baseDir . '/', '', $epub),
                'cover' => $coverPath ?: 'default-cover.png'
            ];
        }
    }

    echo json_encode($books);
} catch (Exception $e) {
    returnError('Failed to process EPUBs: ' . $e->getMessage());
}
?>
