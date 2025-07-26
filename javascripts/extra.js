// Initialize Mermaid diagrams
document.addEventListener('DOMContentLoaded', function() {
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            }
        });
    }
});

// Re-initialize Mermaid on page navigation (for Material theme instant loading)
document.addEventListener('DOMContentLoaded', function() {
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                const mermaidElements = document.querySelectorAll('.mermaid');
                if (mermaidElements.length > 0 && typeof mermaid !== 'undefined') {
                    mermaid.init(undefined, mermaidElements);
                }
            }
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}); 