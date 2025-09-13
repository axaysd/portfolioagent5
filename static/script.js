// Portfolio Management Tool Functions - Designed for LangGraph Integration
window.PortfolioManager = {
    // Portfolio data storage
    portfolio: [],
    availableTags: [], // Track available tags for dynamic columns
    tagDefinitions: {}, // Store specific values for each tag (e.g., {"Asset Class": ["equity", "fixed income", "alternate", "cash"]})
    sessionId: null, // Current session ID
    
    // Initialize portfolio from localStorage
    initialize() {
        // Generate session ID if not exists
        if (!this.sessionId) {
            this.sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }
        
        const saved = localStorage.getItem('portfolio');
        if (saved) {
            const savedData = JSON.parse(saved);
            this.portfolio = savedData.portfolio || savedData; // Handle both old and new formats
            this.availableTags = savedData.tags || []; // Load available tags
            this.tagDefinitions = savedData.tagDefinitions || {}; // Load tag definitions
        }
        
        // Create sample configuration if none exist
        this.createSampleConfigurationIfNeeded();
        
        this.updateUI();
        this.updateWeightSummary();
    },
    
    // Save portfolio to localStorage
    savePortfolio() {
        const portfolioData = {
            portfolio: this.portfolio,
            tags: this.availableTags,
            tagDefinitions: this.tagDefinitions,
            session_id: this.sessionId
        };
        localStorage.setItem('portfolio', JSON.stringify(portfolioData));
    },
    
    // Get current total weight
    getTotalWeight() {
        return this.portfolio.reduce((sum, item) => sum + item.weight, 0);
    },
    
    // Get remaining weight
    getRemainingWeight() {
        return 100 - this.getTotalWeight();
    },
    
    // Add new ticker to portfolio
    addTicker(ticker, weight) {
        // Validate inputs
        if (!ticker || !weight || weight <= 0 || weight > 100) {
            throw new Error('Invalid ticker or weight values');
        }
        
        const tickerUpper = ticker.trim().toUpperCase();
        const weightNum = parseFloat(weight);
        
        // Check if ticker already exists
        if (this.portfolio.some(item => item.ticker === tickerUpper)) {
            throw new Error(`Ticker ${tickerUpper} already exists in portfolio`);
        }
        
        // Check if adding would exceed 100%
        const totalWeight = this.getTotalWeight();
        if (totalWeight + weightNum > 100) {
            throw new Error(`Cannot add ${weightNum}% - would exceed 100% limit. Current total: ${totalWeight.toFixed(2)}%`);
        }
        
        // Add ticker
        this.portfolio.push({ ticker: tickerUpper, weight: weightNum });
        this.savePortfolio();
        this.updateUI();
        this.updateWeightSummary();
        
        return {
            success: true,
            message: `${tickerUpper} added to portfolio with ${weightNum}% weight`,
            portfolio: this.portfolio
        };
    },
    
    // Edit existing ticker
    editTicker(index, newTicker, newWeight) {
        // Validate inputs
        if (index < 0 || index >= this.portfolio.length) {
            throw new Error('Invalid ticker index');
        }
        
        if (!newTicker || !newWeight || newWeight <= 0 || newWeight > 100) {
            throw new Error('Invalid ticker or weight values');
        }
        
        const tickerUpper = newTicker.trim().toUpperCase();
        const weightNum = parseFloat(newWeight);
        
        // Check if new ticker already exists (excluding current item)
        if (this.portfolio.some((item, idx) => idx !== index && item.ticker === tickerUpper)) {
            throw new Error(`Ticker ${tickerUpper} already exists in portfolio`);
        }
        
        // Check if new weight would exceed 100%
        const totalWeight = this.portfolio.reduce((sum, item, idx) => 
            idx !== index ? sum + item.weight : sum, 0);
        
        if (totalWeight + weightNum > 100) {
            throw new Error(`Cannot set weight to ${weightNum}% - would exceed 100% limit. Current total without this ticker: ${totalWeight.toFixed(2)}%`);
        }
        
        // Update ticker
        const oldTicker = this.portfolio[index].ticker;
        this.portfolio[index] = { ticker: tickerUpper, weight: weightNum };
        this.savePortfolio();
        this.updateUI();
        this.updateWeightSummary();
        
        return {
            success: true,
            message: `${oldTicker} updated to ${tickerUpper} with ${weightNum}% weight`,
            portfolio: this.portfolio
        };
    },
    
    // Delete ticker from portfolio
    deleteTicker(index) {
        if (index < 0 || index >= this.portfolio.length) {
            throw new Error('Invalid ticker index');
        }
        
        const ticker = this.portfolio[index].ticker;
        this.portfolio.splice(index, 1);
        this.savePortfolio();
        this.updateUI();
        this.updateWeightSummary();
        
        return {
            success: true,
            message: `${ticker} removed from portfolio`,
            portfolio: this.portfolio
        };
    },
    
    // Get portfolio summary
    getPortfolioSummary() {
        const totalWeight = this.getTotalWeight();
        const remainingWeight = this.getRemainingWeight();
        
        return {
            tickerCount: this.portfolio.length,
            totalWeight: totalWeight.toFixed(2),
            remainingWeight: remainingWeight.toFixed(2),
            isFullyAllocated: totalWeight === 100,
            isOverAllocated: totalWeight > 100,
            tickers: this.portfolio.map(item => ({
                symbol: item.ticker,
                weight: item.weight.toFixed(2)
            }))
        };
    },
    
    // Update UI components
    updateUI() {
        const portfolioTable = document.getElementById('portfolioTable');
        const portfolioTableBody = document.getElementById('portfolioTableBody');
        const emptyState = document.getElementById('emptyState');
        const tableWrapper = document.querySelector('.table-wrapper');
        
        if (this.portfolio.length === 0) {
            portfolioTable.classList.remove('show');
            emptyState.style.display = 'flex';
            if (tableWrapper) tableWrapper.style.display = 'none';
        } else {
            portfolioTable.classList.add('show');
            emptyState.style.display = 'none';
            if (tableWrapper) tableWrapper.style.display = 'block';
            this.renderPortfolioTable();
        }
    },
    
    // Update weight summary display
    updateWeightSummary() {
        const weightSummary = document.getElementById('weightSummary');
        if (!weightSummary) return;
        
        const total = this.getTotalWeight();
        const remaining = this.getRemainingWeight();
        
        const totalElement = weightSummary.querySelector('.weight-total strong');
        const remainingElement = weightSummary.querySelector('.weight-remaining strong');
        
        if (totalElement && remainingElement) {
            totalElement.textContent = `${total.toFixed(2)}%`;
            remainingElement.textContent = `${remaining.toFixed(2)}%`;
            
            // Update colors based on weight status
            if (total > 100) {
                totalElement.style.color = '#dc3545';
            } else if (total === 100) {
                totalElement.style.color = '#28a745';
            } else {
                totalElement.style.color = '#007bff';
            }
            
            if (remaining < 0) {
                remainingElement.style.color = '#dc3545';
            } else if (remaining === 0) {
                remainingElement.style.color = '#28a745';
            } else {
                remainingElement.style.color = '#28a745';
            }
        }
    },
    
    // Render portfolio table
    renderPortfolioTable() {
        const portfolioTableBody = document.getElementById('portfolioTableBody');
        if (!portfolioTableBody) return;
        
        // Update table header with dynamic columns
        this.updateTableHeader();
        
        portfolioTableBody.innerHTML = '';
        
        this.portfolio.forEach((item, index) => {
            const row = document.createElement('tr');
            
            // Build row content with dynamic tag columns
            let rowContent = `
                <td><strong>${item.ticker}</strong></td>
                <td>${item.weight.toFixed(2)}%</td>
            `;
            
            // Add tag columns if they exist
            if (this.availableTags && this.availableTags.length > 0) {
                this.availableTags.forEach(tag => {
                    let tagValue = item[tag] || 'N/A';
                    // Ensure tag value is a string (handle objects/arrays)
                    if (typeof tagValue === 'object' && tagValue !== null) {
                        tagValue = JSON.stringify(tagValue);
                    }
                    rowContent += `<td class="editable-tag-cell" data-ticker="${item.ticker}" data-tag="${tag}" data-original-value="${tagValue}">
                        <span class="tag-display">${tagValue}</span>
                        <input type="text" class="tag-edit-input" value="${tagValue}" style="display: none;" />
                    </td>`;
                });
            }
            
            // Add action buttons
            rowContent += `
                <td class="row-actions">
                    <button class="action-btn edit-btn" onclick="PortfolioManager.openEditModal(${index})" title="Edit ${item.ticker}">✏️</button>
                    <button class="action-btn delete-btn" onclick="PortfolioManager.deleteTickerWithConfirmation(${index})" title="Delete ${item.ticker}">🗑️</button>
                </td>
            `;
            
            row.innerHTML = rowContent;
            portfolioTableBody.appendChild(row);
        });
        
        // Enable tag editing after table is rendered
        this.enableTagEditing();
    },
    
    // Update table header with dynamic tag columns
    updateTableHeader() {
        const headerRow = document.getElementById('portfolioTableHeader');
        if (!headerRow) return;
        
        // Keep Symbol and Weight columns
        let headerContent = `
            <th>Symbol</th>
            <th>Weight (%)</th>
        `;
        
        // Add dynamic tag columns
        if (this.availableTags && this.availableTags.length > 0) {
            this.availableTags.forEach(tag => {
                const tagDisplayName = tag.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                headerContent += `<th>${tagDisplayName}</th>`;
            });
        }
        
        // Add Actions column
        headerContent += `<th>Actions</th>`;
        
        headerRow.innerHTML = headerContent;
    },
    

    
    // Open edit modal
    openEditModal(index) {
        if (index < 0 || index >= this.portfolio.length) return;
        
        const editModal = document.getElementById('editModal');
        const editTickerInput = document.getElementById('editTickerInput');
        const editWeightInput = document.getElementById('editWeightInput');
        
        if (!editModal || !editTickerInput || !editWeightInput) return;
        
        const item = this.portfolio[index];
        editTickerInput.value = item.ticker;
        editWeightInput.value = item.weight;
        
        // Store editing index
        editModal.dataset.editingIndex = index;
        editModal.classList.add('show');
        editTickerInput.focus();
    },
    
    // Save edit changes
    saveEditChanges() {
        const editModal = document.getElementById('editModal');
        const editTickerInput = document.getElementById('editTickerInput');
        const editWeightInput = document.getElementById('editWeightInput');
        
        if (!editModal || !editTickerInput || !editWeightInput) return;
        
        const index = parseInt(editModal.dataset.editingIndex);
        const ticker = editTickerInput.value.trim();
        const weight = parseFloat(editWeightInput.value);
        
        try {
            const result = this.editTicker(index, ticker, weight);
            this.closeEditModal();
            this.showNotification(result.message, 'success');
        } catch (error) {
            this.showNotification(error.message, 'error');
        }
    },
    
    // Close edit modal
    closeEditModal() {
        const editModal = document.getElementById('editModal');
        const editTickerInput = document.getElementById('editTickerInput');
        const editWeightInput = document.getElementById('editWeightInput');
        
        if (editModal) {
            editModal.classList.remove('show');
            delete editModal.dataset.editingIndex;
        }
        
        if (editTickerInput) editTickerInput.value = '';
        if (editWeightInput) editWeightInput.value = '';
    },
    
    // Delete ticker with confirmation
    deleteTickerWithConfirmation(index) {
        if (index < 0 || index >= this.portfolio.length) return;
        
        const ticker = this.portfolio[index].ticker;
        if (confirm(`Are you sure you want to remove ${ticker} from your portfolio?`)) {
            try {
                const result = this.deleteTicker(index);
                this.showNotification(result.message, 'success');
            } catch (error) {
                this.showNotification(error.message, 'error');
            }
        }
    },
    
    // Show notification
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 16px 24px;
            border-radius: 12px;
            color: white;
            font-weight: 500;
            z-index: 1001;
            animation: slideInRight 0.3s ease-out;
            max-width: 350px;
            word-wrap: break-word;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        `;

        // Set background color based on type
        switch (type) {
            case 'success':
                notification.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
                break;
            case 'error':
                notification.style.background = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
                break;
            case 'warning':
                notification.style.background = 'linear-gradient(135deg, #ffc107 0%, #ffb300 100%)';
                notification.style.color = '#212529';
                break;
            default:
                notification.style.background = 'linear-gradient(135deg, #007bff 0%, #0056b3 100%)';
        }

        // Add animation keyframes
        if (!document.querySelector('#notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOutRight {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }

        // Add to DOM
        document.body.appendChild(notification);

        // Remove after 4 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    },

    // Update portfolio from AI response
    updatePortfolioFromAI(aiResponse) {
        // Parse the AI response to extract portfolio changes
        const responseText = aiResponse.toLowerCase();
        console.log('🔍 Parsing AI response:', aiResponse);
        console.log('🔍 Response text (lowercase):', responseText);
        
        let changesMade = false;
        
        // Check if the AI mentioned adding a ticker
        if (responseText.includes('added') || responseText.includes('add')) {
            // Extract ticker and weight from AI response - multiple patterns
            const patterns = [
                /([a-z]{1,5}):\s*(\d+(?:\.\d+)?)%/i,  // TICKER: WEIGHT%
                /added\s+([a-z]{1,5})\s+with\s+(\d+(?:\.\d+)?)%/i,  // added TICKER with WEIGHT%
                /add\s+([a-z]{1,5})\s+(\d+(?:\.\d+)?)%/i,  // add TICKER WEIGHT%
                /added\s+([a-z]{1,5})\s+at\s+(\d+(?:\.\d+)?)%/i,  // added TICKER at WEIGHT%
                /([a-z]{1,5})\s+(\d+(?:\.\d+)?)%/i  // TICKER WEIGHT% (fallback)
            ];
            
            for (const pattern of patterns) {
                const match = responseText.match(pattern);
                if (match) {
                    const ticker = match[1].toUpperCase();
                    const weight = parseFloat(match[2]);
                    console.log('✅ Found ticker to add:', ticker, weight, 'using pattern:', pattern);
                    
                    // Add to portfolio if not already present
                    if (!this.portfolio.some(item => item.ticker === ticker)) {
                        this.addTicker(ticker, weight);
                        this.showNotification(`Added ${ticker} with ${weight}% weight from AI`, 'success');
                        changesMade = true;
                        break;
                    } else {
                        console.log('⚠️ Ticker already exists:', ticker);
                    }
                }
            }
        }
        
        // Check if the AI mentioned modifying a ticker
        if (responseText.includes('modified') || responseText.includes('change') || responseText.includes('modify') || responseText.includes('remains')) {
            const patterns = [
                /([a-z]{1,5}):\s*(\d+(?:\.\d+)?)%/i,  // TICKER: WEIGHT%
                /modified\s+([a-z]{1,5})\s+to\s+(\d+(?:\.\d+)?)%/i,  // modified TICKER to WEIGHT%
                /change\s+([a-z]{1,5})\s+to\s+(\d+(?:\.\d+)?)%/i,  // change TICKER to WEIGHT%
                /([a-z]{1,5})\s+remains\s+(\d+(?:\.\d+)?)%/i,  // TICKER remains WEIGHT%
                /([a-z]{1,5})\s+(\d+(?:\.\d+)?)%/i  // TICKER WEIGHT% (fallback)
            ];
            
            for (const pattern of patterns) {
                const match = responseText.match(pattern);
                if (match) {
                    const ticker = match[1].toUpperCase();
                    const newWeight = parseFloat(match[2]);
                    console.log('✅ Found ticker to modify:', ticker, 'to', newWeight, 'using pattern:', pattern);
                    
                    const index = this.portfolio.findIndex(item => item.ticker === ticker);
                    if (index !== -1) {
                        this.editTicker(index, ticker, newWeight);
                        this.showNotification(`Modified ${ticker} weight to ${newWeight}% as requested by AI`, 'success');
                        changesMade = true;
                        break;
                    } else {
                        console.log('⚠️ Ticker not found for modification:', ticker);
                    }
                }
            }
        }
        
        // Check if the AI mentioned removing a ticker
        if (responseText.includes('removed') || responseText.includes('remove') || responseText.includes('was not removed')) {
            const patterns = [
                /removed\s+([a-z]{1,5})/i,  // removed TICKER
                /([a-z]{1,5})\s+was\s+not\s+removed/i,  // TICKER was not removed
                /([a-z]{1,5})\s+remains/i,  // TICKER remains
                /([a-z]{1,5})\s+not\s+removed/i  // TICKER not removed
            ];
            
            for (const pattern of patterns) {
                const match = responseText.match(pattern);
                if (match) {
                    const ticker = match[1].toUpperCase();
                    console.log('✅ Found ticker for removal check:', ticker, 'using pattern:', pattern);
                    
                    // Check if this is a "was not removed" case - we should remove it manually
                    if (responseText.includes('was not removed') || responseText.includes('not removed')) {
                        const index = this.portfolio.findIndex(item => item.ticker === ticker);
                        if (index !== -1) {
                            this.deleteTicker(index);
                            this.showNotification(`Removed ${ticker} from portfolio as requested by AI`, 'success');
                            changesMade = true;
                            break;
                        }
                    }
                }
            }
        }
        
        if (!changesMade) {
            console.log('⚠️ No portfolio changes detected in AI response');
        }
        
        // Refresh the display
        this.updateUI();
        this.updateWeightSummary();
    },

    // Update portfolio from AI structured data (including tags)
    updatePortfolioFromStructuredData(structuredData) {
        console.log('🔍 Updating portfolio from structured data:', structuredData);
        let changesMade = false;
        
        if (!structuredData || !structuredData.tickers) {
            console.log('⚠️ No valid portfolio data received');
            return;
        }
        
        // Handle portfolio with tags - preserve existing tags and merge new ones
        if (structuredData.tickers && Array.isArray(structuredData.tickers)) {
            // Create a map of existing tickers with their tags
            const existingTickers = new Map();
            this.portfolio.forEach(item => {
                existingTickers.set(item.ticker, item);
            });
            
            // Update portfolio with new ticker data (including tags)
            this.portfolio = structuredData.tickers.map(ticker => {
                const symbol = ticker.symbol || ticker.ticker;
                const newTickerData = {
                    ticker: symbol,
                    weight: parseFloat(ticker.weight)
                };
                
                // Merge with existing tags if this ticker already exists
                const existingTicker = existingTickers.get(symbol);
                if (existingTicker) {
                    // Preserve all existing tags
                    Object.keys(existingTicker).forEach(key => {
                        if (key !== 'ticker' && key !== 'weight') {
                            newTickerData[key] = existingTicker[key];
                        }
                    });
                    console.log(`🔍 Preserved existing tags for ${symbol}:`, existingTicker);
                }
                
                // Add new tags from the AI response
                Object.keys(ticker).forEach(key => {
                    if (key !== 'symbol' && key !== 'ticker' && key !== 'weight') {
                        newTickerData[key] = ticker[key];
                        console.log(`🔍 Added new tag for ${symbol}: ${key} = ${ticker[key]}`);
                    }
                });
                
                return newTickerData;
            });
            
            // Update available tags by collecting all unique tag keys
            const allTags = new Set();
            this.portfolio.forEach(ticker => {
                Object.keys(ticker).forEach(key => {
                    if (key !== 'ticker' && key !== 'weight') {
                        allTags.add(key);
                    }
                });
            });
            this.availableTags = Array.from(allTags);
            console.log('🔍 Updated available tags:', this.availableTags);
            
            changesMade = true;
        }
        
        // Update UI and show notification
        this.updateUI();
        this.updateWeightSummary();
        
        if (structuredData.changes && structuredData.changes.length > 0) {
            const changeMessage = structuredData.changes.join(', ');
            this.showNotification(`Portfolio updated: ${changeMessage}`, 'success');
        } else if (changesMade) {
            this.showNotification('Portfolio updated from AI response', 'success');
        }
        
        console.log('✅ Portfolio updated from structured data with tag preservation');
    },

    // Session Management Methods
    startNewSession() {
        // Generate new session ID
        this.sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        
        // Clear portfolio, tags, and definitions
        this.portfolio = [];
        this.availableTags = [];
        this.tagDefinitions = {};
        
        // Clear chat messages
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
        
        // Save new session state
        this.savePortfolio();
        
        // Update UI
        this.updateUI();
        
        // Show welcome message
        let welcomeMessage = "🆕 New session started! I'm your AI-powered portfolio assistant. How can I help you today?";
        
        // If tags are loaded, mention them
        if (this.availableTags.length > 0) {
            const tagList = this.availableTags.join(', ');
            welcomeMessage += `\n\n📋 Tag configuration loaded: ${tagList}`;
            
            // Show tag definitions if available
            const tagDefinitions = this.getAllTagDefinitions();
            const definedTags = Object.keys(tagDefinitions);
            if (definedTags.length > 0) {
                welcomeMessage += `\n\n🏷️ Tag definitions:`;
                definedTags.forEach(tag => {
                    const values = tagDefinitions[tag].join(', ');
                    welcomeMessage += `\n• ${tag}: ${values}`;
                });
            }
            
            welcomeMessage += `\n\nI'll automatically populate appropriate values for these tags when you add tickers.`;
            welcomeMessage += `\nYou can also define tag values by saying: "tag [tag name] ([value1] vs [value2] vs [value3])"`;
        }
        
        window.ChatManager.addMessage(welcomeMessage, false);
        
        this.showNotification('New session started', 'success');
        console.log('✅ New session started:', this.sessionId);
    },

    // Create sample configuration if none exist
    createSampleConfigurationIfNeeded() {
        const existingConfigs = this.getSavedTagConfigurations();
        if (existingConfigs.length === 0) {
            console.log('🔍 Creating sample tag configuration...');
            const sampleConfig = {
                name: 'Sample Portfolio Tags',
                tags: ['asset_class', 'region'],
                tagDefinitions: {
                    'asset_class': ['Equity', 'Fixed Income', 'Alternative', 'Cash'],
                    'region': ['North America', 'Europe', 'Asia', 'Emerging Markets', 'Global']
                },
                created_at: new Date().toISOString(),
                session_id: this.sessionId
            };
            
            const configs = [sampleConfig];
            localStorage.setItem('portfolio_tag_configs', JSON.stringify(configs));
            console.log('✅ Sample configuration created:', sampleConfig);
        }
    },

    // Tag Configuration Management
    saveTagConfiguration(configName) {
        if (!configName || configName.trim() === '') {
            this.showNotification('Please enter a configuration name', 'error');
            return;
        }
        
        const configNameClean = configName.trim();
        
        // Get saved configurations
        const savedConfigs = this.getSavedTagConfigurations();
        
        // Check if name already exists
        if (savedConfigs.some(config => config.name === configNameClean)) {
            this.showNotification('Configuration name already exists', 'error');
            return;
        }
        
        // Create new configuration
        const newConfig = {
            name: configNameClean,
            tags: [...this.availableTags],
            tagDefinitions: {...this.tagDefinitions},
            created_at: new Date().toISOString(),
            session_id: this.sessionId
        };
        
        // Add to saved configurations
        savedConfigs.push(newConfig);
        
        // Save to localStorage
        localStorage.setItem('portfolio_tag_configs', JSON.stringify(savedConfigs));
        
        this.showNotification(`Tag configuration '${configNameClean}' saved successfully`, 'success');
        console.log('✅ Tag configuration saved:', newConfig);
    },

    loadTagConfiguration(configName) {
        const savedConfigs = this.getSavedTagConfigurations();
        const config = savedConfigs.find(c => c.name === configName);
        
        if (!config) {
            this.showNotification('Configuration not found', 'error');
            return;
        }
        
        // Replace current tags and definitions with loaded configuration
        this.availableTags = [...config.tags];
        this.tagDefinitions = {...(config.tagDefinitions || {})};
        
        // Update UI
        this.updateUI();
        this.savePortfolio();
        
        this.showNotification(`Tag configuration '${configName}' loaded successfully`, 'success');
        console.log('✅ Tag configuration loaded:', config);
        
        // Show tag definitions in chat
        const tagDefinitions = this.getAllTagDefinitions();
        const definedTags = Object.keys(tagDefinitions);
        if (definedTags.length > 0) {
            let definitionMessage = `🏷️ Tag definitions loaded:`;
            definedTags.forEach(tag => {
                const values = tagDefinitions[tag].join(', ');
                definitionMessage += `\n• ${tag}: ${values}`;
            });
            window.ChatManager.addMessage(definitionMessage, false);
        }
        
        // If there are existing tickers, ask AI to populate tag values
        if (this.portfolio.length > 0 && this.availableTags.length > 0) {
            const tickerList = this.portfolio.map(item => item.ticker).join(', ');
            const tagList = this.availableTags.join(', ');
            
            // Include tag definitions in the message
            let message = `I've loaded the tag configuration "${configName}" with tags: ${tagList}.`;
            
            // Add tag definitions if available
            const tagDefinitions = this.getAllTagDefinitions();
            const definedTags = Object.keys(tagDefinitions);
            if (definedTags.length > 0) {
                message += `\n\nTag definitions:`;
                definedTags.forEach(tag => {
                    const values = tagDefinitions[tag].join(', ');
                    message += `\n• ${tag}: ${values}`;
                });
            }
            
            message += `\n\nPlease populate the appropriate values for these tags for my existing tickers: ${tickerList}.`;
            
            // Add a small delay to ensure UI is updated first
            setTimeout(() => {
                window.ChatManager.sendMessage(message);
            }, 500);
        }
    },

    getSavedTagConfigurations() {
        const saved = localStorage.getItem('portfolio_tag_configs');
        return saved ? JSON.parse(saved) : [];
    },

    deleteTagConfiguration(configName) {
        const savedConfigs = this.getSavedTagConfigurations();
        const filteredConfigs = savedConfigs.filter(config => config.name !== configName);
        
        localStorage.setItem('portfolio_tag_configs', JSON.stringify(filteredConfigs));
        
        this.showNotification(`Configuration '${configName}' deleted`, 'success');
        console.log('✅ Tag configuration deleted:', configName);
    },

    // Tag Definition Management
    setTagDefinition(tagName, values) {
        if (!tagName || !values || !Array.isArray(values) || values.length === 0) {
            this.showNotification('Invalid tag definition', 'error');
            return;
        }
        
        // Clean and validate values
        const cleanValues = values.map(v => v.trim()).filter(v => v.length > 0);
        
        if (cleanValues.length === 0) {
            this.showNotification('No valid values provided', 'error');
            return;
        }
        
        this.tagDefinitions[tagName] = cleanValues;
        this.savePortfolio();
        
        this.showNotification(`Tag definition for '${tagName}' updated`, 'success');
        console.log('✅ Tag definition set:', tagName, cleanValues);
    },

    // Tag Value Editing Functions
    enableTagEditing() {
        // Add click handlers to all editable tag cells
        const editableCells = document.querySelectorAll('.editable-tag-cell');
        editableCells.forEach(cell => {
            cell.addEventListener('click', this.handleTagCellClick.bind(this));
        });
    },

    handleTagCellClick(event) {
        const cell = event.currentTarget;
        const displaySpan = cell.querySelector('.tag-display');
        const inputField = cell.querySelector('.tag-edit-input');
        
        // Hide display span and show input field
        displaySpan.style.display = 'none';
        inputField.style.display = 'block';
        inputField.focus();
        inputField.select();
        
        // Add event listeners for save/cancel
        inputField.addEventListener('blur', this.handleTagEditBlur.bind(this));
        inputField.addEventListener('keydown', this.handleTagEditKeydown.bind(this));
    },

    handleTagEditBlur(event) {
        this.saveTagEdit(event.target);
    },

    handleTagEditKeydown(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            this.saveTagEdit(event.target);
        } else if (event.key === 'Escape') {
            event.preventDefault();
            this.cancelTagEdit(event.target);
        }
    },

    saveTagEdit(inputField) {
        const cell = inputField.closest('.editable-tag-cell');
        const displaySpan = cell.querySelector('.tag-display');
        const ticker = cell.dataset.ticker;
        const tag = cell.dataset.tag;
        const newValue = inputField.value.trim();
        const originalValue = cell.dataset.originalValue;
        
        // Update the portfolio data
        const tickerIndex = this.portfolio.findIndex(item => item.ticker === ticker);
        if (tickerIndex !== -1) {
            this.portfolio[tickerIndex][tag] = newValue || 'N/A';
            this.savePortfolio();
            
            // Update display
            displaySpan.textContent = newValue || 'N/A';
            cell.dataset.originalValue = newValue || 'N/A';
            
            this.showNotification(`Updated ${tag} for ${ticker} to "${newValue || 'N/A'}"`, 'success');
        }
        
        // Hide input and show display
        inputField.style.display = 'none';
        displaySpan.style.display = 'inline';
        
        // Remove event listeners
        inputField.removeEventListener('blur', this.handleTagEditBlur.bind(this));
        inputField.removeEventListener('keydown', this.handleTagEditKeydown.bind(this));
    },

    cancelTagEdit(inputField) {
        const cell = inputField.closest('.editable-tag-cell');
        const displaySpan = cell.querySelector('.tag-display');
        const originalValue = cell.dataset.originalValue;
        
        // Restore original value
        inputField.value = originalValue;
        
        // Hide input and show display
        inputField.style.display = 'none';
        displaySpan.style.display = 'inline';
        
        // Remove event listeners
        inputField.removeEventListener('blur', this.handleTagEditBlur.bind(this));
        inputField.removeEventListener('keydown', this.handleTagEditKeydown.bind(this));
    },

    getTagDefinition(tagName) {
        return this.tagDefinitions[tagName] || [];
    },

    getAllTagDefinitions() {
        return this.tagDefinitions;
    },

    clearTagDefinition(tagName) {
        if (this.tagDefinitions[tagName]) {
            delete this.tagDefinitions[tagName];
            this.savePortfolio();
            this.showNotification(`Tag definition for '${tagName}' cleared`, 'success');
        }
    },

    // Parse AI response for tag definition commands
    parseTagDefinitionCommand(message) {
        const lowerMessage = message.toLowerCase();
        
        // Look for patterns like "tag asset classes (equity vs fixed income vs alternate vs cash)"
        const tagDefinitionPattern = /tag\s+([^(]+)\s*\(([^)]+)\)/i;
        const match = message.match(tagDefinitionPattern);
        
        if (match) {
            const tagName = match[1].trim();
            const valuesString = match[2].trim();
            
            // Split by common separators and clean up
            const values = valuesString
                .split(/vs\.?|,|;|\|/)
                .map(v => v.trim())
                .filter(v => v.length > 0);
            
            if (values.length > 0) {
                this.setTagDefinition(tagName, values);
                
                // Show notification
                this.showNotification(`Tag definition for '${tagName}' updated: ${values.join(', ')}`);
                
                return true;
            }
        }
        
        return false;
    },
};

// Tag Configuration Manager
window.TagConfigManager = {
    currentMode: null, // 'save' or 'load'
    selectedConfig: null,

    openSaveModal() {
        this.currentMode = 'save';
        const modal = document.getElementById('tagConfigModal');
        const title = document.getElementById('tagConfigTitle');
        const saveContent = document.getElementById('saveTagConfigContent');
        const loadContent = document.getElementById('loadTagConfigContent');
        const configInput = document.getElementById('configNameInput');
        const currentTagsList = document.getElementById('currentTagsList');

        title.textContent = 'Save Tag Configuration';
        saveContent.style.display = 'block';
        loadContent.style.display = 'none';
        
        // Clear input
        configInput.value = '';
        
        // Show current tags
        this.displayCurrentTags(currentTagsList);
        
        modal.classList.add('show');
        configInput.focus();
    },

    openLoadModal() {
        console.log('🔍 Opening load modal...');
        this.currentMode = 'load';
        const modal = document.getElementById('tagConfigModal');
        const title = document.getElementById('tagConfigTitle');
        const saveContent = document.getElementById('saveTagConfigContent');
        const loadContent = document.getElementById('loadTagConfigContent');
        const savedConfigsList = document.getElementById('savedConfigsList');

        console.log('🔍 Modal elements found:', { modal, title, saveContent, loadContent, savedConfigsList });

        title.textContent = 'Load Tag Configuration';
        saveContent.style.display = 'none';
        loadContent.style.display = 'block';
        
        // Show saved configurations
        this.displaySavedConfigurations(savedConfigsList);
        
        modal.classList.add('show');
        console.log('🔍 Modal should now be visible');
    },

    displayCurrentTags(container) {
        const tags = window.PortfolioManager.availableTags;
        const tagDefinitions = window.PortfolioManager.getAllTagDefinitions();
        
        if (tags.length === 0) {
            container.innerHTML = '<p style="color: #6c757d; font-style: italic;">No tags configured yet</p>';
            return;
        }
        
        container.innerHTML = tags.map(tag => {
            const definitions = tagDefinitions[tag];
            const definitionText = definitions ? ` (${definitions.join(', ')})` : '';
            return `<div class="tag-item">${tag}${definitionText}</div>`;
        }).join('');
    },

    displaySavedConfigurations(container) {
        console.log('🔍 Displaying saved configurations...');
        const configs = window.PortfolioManager.getSavedTagConfigurations();
        console.log('🔍 Found configurations:', configs);
        
        if (configs.length === 0) {
            console.log('🔍 No configurations found, showing empty message');
            container.innerHTML = '<p style="color: #6c757d; font-style: italic;">No saved configurations found</p>';
            return;
        }
        
        container.innerHTML = configs.map(config => {
            const date = new Date(config.created_at).toLocaleDateString();
            const tagDefinitions = config.tagDefinitions || {};
            const hasDefinitions = Object.keys(tagDefinitions).length > 0;
            const definitionText = hasDefinitions ? ' • With definitions' : '';
            
            return `
                <div class="saved-config-item" data-config-name="${config.name}">
                    <div class="config-name">${config.name}</div>
                    <div class="config-details">
                        ${config.tags.length} tags${definitionText} • Created ${date}
                    </div>
                </div>
            `;
        }).join('');
    },

    closeModal() {
        const modal = document.getElementById('tagConfigModal');
        modal.classList.remove('show');
        this.currentMode = null;
        this.selectedConfig = null;
    },

    confirmAction() {
        if (this.currentMode === 'save') {
            const configName = document.getElementById('configNameInput').value;
            window.PortfolioManager.saveTagConfiguration(configName);
        } else if (this.currentMode === 'load' && this.selectedConfig) {
            window.PortfolioManager.loadTagConfiguration(this.selectedConfig);
            
            // Add helpful message about AI populating values
            const config = window.PortfolioManager.getSavedTagConfigurations().find(c => c.name === this.selectedConfig);
            if (config && config.tags.length > 0) {
                const tagList = config.tags.join(', ');
                let message = `📋 Tag configuration "${this.selectedConfig}" loaded with tags: ${tagList}.`;
                
                // Show tag definitions if available
                const tagDefinitions = config.tagDefinitions || {};
                const definedTags = Object.keys(tagDefinitions);
                if (definedTags.length > 0) {
                    message += `\n\n🏷️ Tag definitions:`;
                    definedTags.forEach(tag => {
                        const values = tagDefinitions[tag].join(', ');
                        message += `\n• ${tag}: ${values}`;
                    });
                }
                
                message += `\n\nI'll automatically populate appropriate values for these tags when you add tickers.`;
                window.ChatManager.addMessage(message, false);
            }
        }
        
        this.closeModal();
    }
};

// Chat Manager for LangGraph AI Integration
window.ChatManager = {
    // Conversation history storage
    conversationHistory: [],
    
    // Add a message to the chat
    addMessage(content, isUser = false) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        // Store in conversation history
        this.conversationHistory.push({
            content: content,
            isUser: isUser,
            timestamp: new Date().toISOString()
        });
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    },

    // Send a message and get AI response
    async sendMessage(message) {
        if (!message.trim()) return;
        
        // Check for tag definition commands first
        const wasTagDefinition = window.PortfolioManager.parseTagDefinitionCommand(message);
        
        // Add user message
        this.addMessage(message, true);
        
        // Get current portfolio data
        const portfolio = window.PortfolioManager.getPortfolioSummary();
        
        try {
            // Show loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message bot-message loading';
            loadingDiv.innerHTML = '<div class="message-content">🤔 Thinking...</div>';
            document.getElementById('chatMessages').appendChild(loadingDiv);
            
            // Send to LangGraph backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    portfolio: portfolio,
                    available_tags: window.PortfolioManager.availableTags,
                    tag_definitions: window.PortfolioManager.getAllTagDefinitions(),
                    conversation_history: this.conversationHistory
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Remove loading indicator
            loadingDiv.remove();
            
            if (data.success) {
                // Add AI response
                this.addMessage(data.response, false);
                
                // Update portfolio with structured data from backend
                if (data.portfolio) {
                    window.PortfolioManager.updatePortfolioFromStructuredData(data.portfolio);
                }
                
                // Update available tags and tag definitions from backend
                if (data.available_tags) {
                    window.PortfolioManager.availableTags = data.available_tags;
                }
                if (data.tag_definitions) {
                    window.PortfolioManager.tagDefinitions = data.tag_definitions;
                }
                
                // Save updated portfolio state
                window.PortfolioManager.savePortfolio();
                
                // Log changes for debugging
                if (data.changes) {
                    console.log('🔍 AI Changes:', data.changes);
                }
            } else {
                this.addMessage("Sorry, I encountered an error processing your request. Please try again.", false);
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            
            // Remove loading indicator
            const loadingDiv = document.querySelector('.loading');
            if (loadingDiv) loadingDiv.remove();
            
            // Provide more specific error messages
            let errorMessage = "Sorry, I encountered an error processing your request.";
            
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                errorMessage = "Network error: Please check your internet connection and try again.";
            } else if (error.message.includes('HTTP error')) {
                errorMessage = `Server error: ${error.message}. Please try again later.`;
            } else {
                errorMessage = `Error: ${error.message}. Please try again.`;
            }
            
            this.addMessage(errorMessage, false);
        }
    },

    // Initialize chat with welcome message
    initialize() {
        this.addMessage("Hello! I'm your AI-powered portfolio assistant. I can help you manage your portfolio, add/remove tickers, and provide insights. How can I help you today?", false);
    }
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize portfolio manager
    PortfolioManager.initialize();
    
    // Initialize chat manager
    ChatManager.initialize();
    
    // Setup event listeners
    setupEventListeners();
    
    // Focus on chat input
    const messageInput = document.getElementById('messageInput');
    if (messageInput) messageInput.focus();
});

// Setup all event listeners
function setupEventListeners() {
    // Session management event listeners
    const newSessionBtn = document.getElementById('newSessionBtn');
    const saveTagsBtn = document.getElementById('saveTagsBtn');
    const loadTagsBtn = document.getElementById('loadTagsBtn');
    
    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', () => {
            if (confirm('Start a new session? This will clear your current portfolio and chat history.')) {
                window.PortfolioManager.startNewSession();
            }
        });
    }
    
    if (saveTagsBtn) {
        saveTagsBtn.addEventListener('click', () => {
            window.TagConfigManager.openSaveModal();
        });
    }
    
    if (loadTagsBtn) {
        loadTagsBtn.addEventListener('click', () => {
            console.log('🔍 Load tags button clicked');
            window.TagConfigManager.openLoadModal();
        });
    } else {
        console.log('❌ Load tags button not found');
    }

    // Tag configuration modal event listeners
    const tagConfigModal = document.getElementById('tagConfigModal');
    const closeTagConfigModal = document.getElementById('closeTagConfigModal');
    const cancelTagConfig = document.getElementById('cancelTagConfig');
    const confirmTagConfig = document.getElementById('confirmTagConfig');
    
    if (closeTagConfigModal) {
        closeTagConfigModal.addEventListener('click', () => {
            window.TagConfigManager.closeModal();
        });
    }
    
    if (cancelTagConfig) {
        cancelTagConfig.addEventListener('click', () => {
            window.TagConfigManager.closeModal();
        });
    }
    
    if (confirmTagConfig) {
        confirmTagConfig.addEventListener('click', () => {
            window.TagConfigManager.confirmAction();
        });
    }
    
    // Close modal when clicking outside
    if (tagConfigModal) {
        tagConfigModal.addEventListener('click', (e) => {
            if (e.target === tagConfigModal) {
                window.TagConfigManager.closeModal();
            }
        });
    }
    
    // Handle saved configuration selection
    document.addEventListener('click', (e) => {
        if (e.target.closest('.saved-config-item')) {
            const configItem = e.target.closest('.saved-config-item');
            const configName = configItem.dataset.configName;
            
            // Remove previous selection
            document.querySelectorAll('.saved-config-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            // Select current item
            configItem.classList.add('selected');
            window.TagConfigManager.selectedConfig = configName;
        }
    });

    // Chat event listeners
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    if (messageInput) {
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const message = messageInput.value.trim();
                if (message) {
                    ChatManager.sendMessage(message);
                    messageInput.value = '';
                }
            }
        });
    }
    
    if (sendButton) {
        sendButton.addEventListener('click', () => {
            const message = messageInput.value.trim();
            if (message) {
                ChatManager.sendMessage(message);
                messageInput.value = '';
            }
        });
    }

    // Portfolio table event listeners (handled by PortfolioManager methods)
    // These are set up in renderPortfolioTable() when the table is rendered
}


