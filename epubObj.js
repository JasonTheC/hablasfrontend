export class EpubTocExtractor {
    constructor() {
        this.zip = null;
    }

    async extractToc(epubFile) {
        // Load the EPUB file as a ZIP
        const zip = await JSZip.loadAsync(epubFile);
        this.zip = zip;
        console.log("1. extracting epub as a zip")

        // First try to find the TOC using the modern nav.xhtml approach
        const containerXml = await this.getContainerXml();
        const rootFilePath = this.parseContainerXml(containerXml);
        const opfContent = await this.getOpfContent(rootFilePath);
        const navPath = this.findNavPath(opfContent);

        if (navPath) {
            const navContent = await this.getFileContent(navPath);
            return this.parseNavDocument(navContent);
        }

        // Fallback to NCX if nav.xhtml is not found
        const ncxPath = this.findNcxPath(opfContent);
        if (ncxPath) {
            const ncxContent = await this.getFileContent(ncxPath);
            return this.parseNcxDocument(ncxContent);
        }

        throw new Error('No table of contents found');
    }

    async getContainerXml() {
        console.log("2.getting file content")
        return await this.getFileContent('META-INF/container.xml');

    }

    async getFileContent(path) {
        const file = this.zip.file(path);
        console.log("3.getting file content PATH")
        if (!file) throw new Error(`File not found: ${path}`);
        return await file.async('text');
    }

    parseContainerXml(containerXml) {
        const parser = new DOMParser();
        console.log("4.container XML")
        const doc = parser.parseFromString(containerXml, 'text/xml');
        const rootFilePath = doc.querySelector('rootfile').getAttribute('full-path');
        return rootFilePath;
    }

    async getOpfContent(rootFilePath) {
        return await this.getFileContent(rootFilePath);
    }

    findNavPath(opfContent) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(opfContent, 'text/xml');
        const navItem = doc.querySelector('item[properties~="nav"]');
        return navItem ? navItem.getAttribute('href') : null;
    }

    findNcxPath(opfContent) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(opfContent, 'text/xml');
        const ncxItem = doc.querySelector('item[media-type="application/x-dtbncx+xml"]');
        return ncxItem ? ncxItem.getAttribute('href') : null;
    }

    parseNavDocument(navContent) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(navContent, 'text/html');
        const nav = doc.querySelector('nav[*|type="toc"]') || doc.querySelector('nav');
        
        if (!nav) return [];
        
        const extractItems = (element) => {
            const items = [];
            const lis = element.querySelectorAll('li');
            
            lis.forEach(li => {
                const a = li.querySelector('a');
                if (a) {
                    const item = {
                        label: a.textContent.trim(),
                        href: a.getAttribute('href'),
                        children: []
                    };
                    
                    const nestedOl = li.querySelector('ol');
                    if (nestedOl) {
                        item.children = extractItems(nestedOl);
                    }
                    
                    items.push(item);
                }
            });
            
            return items;
        };

        console.log("5. navcontent in js obj")
        return extractItems(nav);
    }

    parseNcxDocument(ncxContent) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(ncxContent, 'text/xml');
        
        const extractItems = (navPoints) => {
            const items = [];
            
            navPoints.forEach(navPoint => {
                const labelEl = navPoint.querySelector('navLabel text');
                const contentEl = navPoint.querySelector('content');
                
                if (labelEl && contentEl) {
                    const item = {
                        label: labelEl.textContent.trim(),
                        href: contentEl.getAttribute('src'),
                        children: []
                    };
                    
                    const nestedNavPoints = navPoint.querySelectorAll(':scope > navPoint');
                    if (nestedNavPoints.length > 0) {
                        item.children = extractItems(Array.from(nestedNavPoints));
                    }
                    
                    items.push(item);
                }
            });
            console.log("6. we're parsing in epubObj.js")
            return items;
        };

        const navPoints = Array.from(doc.querySelectorAll('navMap > navPoint'));
        return extractItems(navPoints);
    }
}

// Example usage:
export async function displayToc(epubFile) {
    try {
        const extractor = new EpubTocExtractor();
        const toc = await extractor.extractToc(epubFile);
        console.log("displayToc async function")
        const renderTocItem = (item, level = 0) => {
            const div = document.createElement('div');
            div.style.paddingLeft = `${level * 20}px`;
            
            const link = document.createElement('a');
            link.href = item.href;
            link.textContent = item.label;
            div.appendChild(link);
            
            item.children.forEach(child => {
                div.appendChild(renderTocItem(child, level + 1));
            });
            
            return div;
        };
        
        const tocContainer = document.createElement('div');
        tocContainer.className = 'toc-container';
        toc.forEach(item => {
            tocContainer.appendChild(renderTocItem(item));
        });
        
        return tocContainer

    } catch (error) {
        console.error('Error extracting TOC:', error);
    }
}

// When handling a file input
//fileInput.addEventListener('change', async (e) => {
//    const file = e.target.files[0];
//`    await displayToc(file);
//});


//[
//    {
//        label: "Chapter 1",
//        href: "chapter1.xhtml",
//        children: [
//            {
//                label: "Section 1.1",
//                href: "chapter1.xhtml#section1",
//                children: []
//            }
//        ]
//    }
    // ... more chapters
//]