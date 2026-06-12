/* ==========================================================================
   Frontend JavaScript — Stock Analyser Pro
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // API Endpoints
    const ANALYZE_API = '/api/analyze';
    const STATUS_API = '/api/status';

    // Application Views
    const searchSection = document.getElementById('search-section');
    const chooserSection = document.getElementById('chooser-section');
    const loadingSection = document.getElementById('loading-section');
    const dashboardSection = document.getElementById('dashboard-section');
    
    // Search Form elements
    const searchForm = document.getElementById('search-form');
    const stockInput = document.getElementById('stock-input');
    const searchBtn = document.getElementById('search-btn');
    const suggestBtns = document.querySelectorAll('.suggest-btn');
    
    // Header Info
    const headerMeta = document.getElementById('header-meta');
    const metaTicker = document.getElementById('meta-ticker');
    const metaName = document.getElementById('meta-name');
    
    // Dashboard Core Info
    const stockTitleFull = document.getElementById('stock-title-full');
    const stockTickerVal = document.getElementById('stock-ticker-val');
    const stockExchangeVal = document.getElementById('stock-exchange-val');
    const summaryRecBadge = document.getElementById('summary-rec-badge');
    const quickConviction = document.getElementById('quick-conviction');
    const quickRisk = document.getElementById('quick-risk');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');

    // Loading View Elements
    const loadingTitle = document.getElementById('loading-title');
    const loadingSubtitle = document.getElementById('loading-subtitle');
    const terminalBody = document.getElementById('terminal-body');
    const stepPrefetch = document.getElementById('step-prefetch');
    const stepFundamental = document.getElementById('step-fundamental');
    const stepTechnical = document.getElementById('step-technical');
    const stepSentiment = document.getElementById('step-sentiment');
    const stepSynthesis = document.getElementById('step-synthesis');

    // Chart Instance
    let sentimentChart = null;

    // Active polling details
    let pollingInterval = null;
    let loggedLinesCount = 0;

    // Suggestion Buttons
    suggestBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            stockInput.value = btn.getAttribute('data-value');
            stockInput.focus();
        });
    });

    // Form Submission
    searchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = stockInput.value.trim();
        if (!query) return;

        startAnalysisFlow(query);
    });

    // Reset Dashboard and go back to Search
    newAnalysisBtn.addEventListener('click', () => {
        resetAppToSearch();
    });

    // Logo click navigates home from any view
    document.getElementById('logo-home-btn').addEventListener('click', () => {
        resetAppToSearch();
    });

    // Setup Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(`tab-${tabId}`).classList.add('active');
        });
    });

    // Listing Chooser — shown when a company is dual-listed on India + US
    const chooserTitle = document.getElementById('chooser-title');
    const chooserSubtitle = document.getElementById('chooser-subtitle');
    const chooserOptions = document.getElementById('chooser-options');
    document.getElementById('chooser-back-btn').addEventListener('click', () => resetAppToSearch());

    function showListingChooser(options) {
        const companyName = options[0].full_name.split(' ').slice(0, 3).join(' ');
        chooserTitle.innerText = `${companyName} — Choose a Listing`;
        chooserSubtitle.innerText = 'This company is listed on multiple exchanges. Select which listing you want to analyse.';

        chooserOptions.innerHTML = '';
        options.forEach(opt => {
            const isIndian = opt.ticker.endsWith('.NS') || opt.ticker.endsWith('.BO');
            const flag = isIndian ? '🇮🇳' : '🇺🇸';
            const card = document.createElement('div');
            card.className = 'chooser-option-card';
            card.innerHTML = `
                <div class="chooser-flag">${flag}</div>
                <div class="chooser-exchange">${escapeHtml(opt.exchange_name)}</div>
                <div class="chooser-ticker">${escapeHtml(opt.ticker)}</div>
                <div class="chooser-name">${escapeHtml(opt.full_name)}</div>
                <button class="btn btn-primary btn-small chooser-select-btn">
                    <span>Analyse this listing</span>
                    <i class="fa-solid fa-arrow-right btn-icon"></i>
                </button>`;
            card.querySelector('.chooser-select-btn').addEventListener('click', () => {
                searchSection.style.display = 'none';
                chooserSection.style.display = 'none';
                startAnalysisFlow(opt.ticker);
            });
            chooserOptions.appendChild(card);
        });

        searchSection.style.display = 'none';
        chooserSection.style.display = 'flex';
        loadingSection.style.display = 'none';
        dashboardSection.style.display = 'none';
        headerMeta.style.display = 'none';
    }

    // Start Analysis Request
    function startAnalysisFlow(query) {
        searchBtn.disabled = true;
        searchBtn.querySelector('span').innerText = 'Resolving...';

        fetch(`${ANALYZE_API}?stock=${encodeURIComponent(query)}`)
            .then(res => {
                if (!res.ok) {
                    return res.json().then(err => { throw new Error(err.detail || 'Ticker resolution failed'); });
                }
                return res.json();
            })
            .then(data => {
                if (data.status === 'choose') {
                    searchBtn.disabled = false;
                    searchBtn.querySelector('span').innerText = 'Analyze Stock';
                    showListingChooser(data.options);
                } else {
                    // status === 'started' (or legacy response without status field)
                    initiateLoadingView(data);
                    pollStatus(data.analysis_id);
                }
            })
            .catch(err => {
                showErrorCard(err.message);
                searchBtn.disabled = false;
                searchBtn.querySelector('span').innerText = 'Analyze Stock';
            });
    }

    function initiateLoadingView(info) {
        // Show loading screen, hide search
        searchSection.style.display = 'none';
        loadingSection.style.display = 'block';
        dashboardSection.style.display = 'none';
        headerMeta.style.display = 'none';

        // Configure loading titles
        loadingTitle.innerText = `Analyzing ${info.company_name}`;
        loadingSubtitle.innerText = `Preparing multi-agent research nodes for ${info.ticker}...`;

        // Clear terminal & reset step states
        terminalBody.innerHTML = `<div class="terminal-line system">[System] Connection established. Analysis ID: ${info.analysis_id}</div>`;
        terminalBody.innerHTML += `<div class="terminal-line">[System] Dispatching agents to research ${info.company_name} (${info.ticker})...</div>`;
        loggedLinesCount = 0;

        resetStepItems();
        stepPrefetch.classList.add('active');
    }

    function resetStepItems() {
        const steps = [stepPrefetch, stepFundamental, stepTechnical, stepSentiment, stepSynthesis];
        steps.forEach(step => {
            step.className = 'step-item';
        });
    }

    // Polling Status Loop
    function pollStatus(analysisId) {
        pollingInterval = setInterval(() => {
            fetch(`${STATUS_API}/${analysisId}`)
                .then(res => res.json())
                .then(data => {
                    updateTerminalAndSteps(data);
                    
                    if (data.status === 'completed') {
                        clearInterval(pollingInterval);
                        displayResults(data);
                    } else if (data.status === 'failed') {
                        clearInterval(pollingInterval);
                        const rawErr = data.error || 'Unknown error';
                        const isRateLimit = rawErr.toLowerCase().includes('ratelimit') || rawErr.toLowerCase().includes('rate limit');
                        const msg = isRateLimit
                            ? 'Groq API rate limit reached. The model has a 12,000 tokens/min cap on the free tier. Please wait 60–90 seconds and try again.'
                            : rawErr.length > 200 ? rawErr.slice(0, 200) + '…' : rawErr;
                        resetAppToSearch();
                        showErrorCard(msg);
                    }
                })
                .catch(err => {
                    console.error('Polling error:', err);
                });
        }, 1500);
    }

    function updateTerminalAndSteps(data) {
        const logs = data.logs || [];
        const status = data.status;

        // 1. Update terminal logs
        if (logs.length > loggedLinesCount) {
            for (let i = loggedLinesCount; i < logs.length; i++) {
                const line = logs[i];
                let cssClass = '';
                
                // Categorize logs for styling
                if (line.includes('[System]') || line.includes('[Backend]')) {
                    cssClass = 'system';
                } else if (line.includes('completed') || line.includes('finished') || line.includes('✓')) {
                    cssClass = 'success';
                } else if (line.includes('failed') || line.includes('Error') || line.includes('Warning')) {
                    cssClass = 'error';
                } else if (line.includes('[LangGraph]')) {
                    cssClass = 'system';
                } else if (line.startsWith('>') || line.includes('Thinking Process:')) {
                    cssClass = 'thought';
                }

                terminalBody.innerHTML += `<div class="terminal-line ${cssClass}">${escapeHtml(line)}</div>`;
            }
            loggedLinesCount = logs.length;
            terminalBody.scrollTop = terminalBody.scrollHeight;
        }

        // 2. Drive active agent steps indicator based on logs content
        const logsText = logs.join('\n');
        
        // Data Prefetch step
        if (logsText.includes('Pre-fetching verified market data')) {
            markStepActive(stepPrefetch);
        }
        if (logsText.includes('Verified live market data pre-fetch completed')) {
            markStepCompleted(stepPrefetch);
        }

        // Fundamental agent step
        if (logsText.includes('starting Fundamental Agent')) {
            markStepActive(stepFundamental);
            loadingSubtitle.innerText = "Fundamental Analyst: Reviewing balance sheets, profit statements, and peers...";
        }
        if (logsText.includes('fundamental_node: complete')) {
            markStepCompleted(stepFundamental);
        }

        // Technical agent step
        if (logsText.includes('starting Technical Agent')) {
            markStepActive(stepTechnical);
            loadingSubtitle.innerText = "Technical Analyst: Evaluating price history, RSI, MACD crossovers, and support/resistance...";
        }
        if (logsText.includes('technical_node: complete')) {
            markStepCompleted(stepTechnical);
        }

        // Sentiment agent step
        if (logsText.includes('starting Sentiment Agent')) {
            markStepActive(stepSentiment);
            loadingSubtitle.innerText = "Sentiment Analyst: Parsing news sentiment scores, insider transactions, and institutional stakes...";
        }
        if (logsText.includes('sentiment_node: complete')) {
            markStepCompleted(stepSentiment);
        }

        // Synthesis agent step
        if (logsText.includes('starting Synthesis Agent')) {
            markStepActive(stepSynthesis);
            loadingSubtitle.innerText = "Chief Investment Strategist: Triangulating metrics into final recommendation...";
        }
        if (logsText.includes('synthesis_node: complete')) {
            markStepCompleted(stepSynthesis);
        }

    }

    function markStepActive(stepElement) {
        if (!stepElement.classList.contains('completed')) {
            stepElement.classList.add('active');
        }
    }

    function markStepCompleted(stepElement) {
        stepElement.classList.remove('active');
        stepElement.classList.add('completed');
    }

    // Display Analysis Results
    function displayResults(data) {
        const results = data.results;
        if (!results) return;

        // Hide loading, show dashboard
        loadingSection.style.display = 'none';
        dashboardSection.style.display = 'block';

        // Set up Header metadata
        headerMeta.style.display = 'flex';
        metaTicker.innerText = data.ticker;
        metaName.innerText = data.company_name;

        // Set up Title block
        stockTitleFull.innerText = data.company_name;
        stockTickerVal.innerText = data.ticker;
        // Check if exchange can be guessed or parsed from the ticker
        stockExchangeVal.innerText = data.ticker.includes('.NS') || data.ticker.includes('.BO') ? 'NSE/BSE (India)' : 'NASDAQ/NYSE (US)';

        // Render Markdown reports using Marked.js
        document.getElementById('text-summary').innerHTML = marked.parse(results.synthesis || '');
        document.getElementById('text-fundamental').innerHTML = marked.parse(results.fundamental || '');
        document.getElementById('text-technical').innerHTML = marked.parse(results.technical || '');
        document.getElementById('text-sentiment').innerHTML = marked.parse(results.sentiment || '');

        // Render Risk Raw data block
        document.getElementById('text-risk-raw').innerText = results.risk_ctx || 'N/A';


        // Parse metrics from text reports to display dynamically
        parseAndPopulateMetrics(results);

        // Scroll to the top of dashboard
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Parse specific values using Regex
    function parseAndPopulateMetrics(results) {
        const synthesis = results.synthesis || '';
        const fundamental = results.fundamental || '';
        const technical = results.technical || '';
        const sentiment = results.sentiment || '';
        const risk_ctx = results.risk_ctx || '';
        const fund_ctx = results.fund_ctx || '';
        const tech_ctx = results.tech_ctx || '';

        // 1. Recommendation
        let rec = 'HOLD';
        if (/RECOMMENDATION:\s*BUY/i.test(synthesis)) {
            rec = 'BUY';
        } else if (/RECOMMENDATION:\s*SELL/i.test(synthesis)) {
            rec = 'SELL';
        }
        summaryRecBadge.innerText = rec;
        summaryRecBadge.className = `rec-badge ${rec.toLowerCase()}`;

        // 2. Conviction and Risk in Summary Bar
        const convictionMatch = synthesis.match(/Conviction Level[:\s]+([A-Za-z]+)/i);
        const quickConvictionVal = convictionMatch ? convictionMatch[1].toUpperCase() : 'MEDIUM';
        quickConviction.innerText = quickConvictionVal;
        quickConviction.className = `q-value ${quickConvictionVal === 'HIGH' ? 'text-gold' : (quickConvictionVal === 'LOW' ? 'text-red' : 'text-orange')}`;

        const riskMatch = synthesis.match(/Risk Level[:\s]+([A-Za-z]+)/i);
        const quickRiskVal = riskMatch ? riskMatch[1].toUpperCase() : 'MEDIUM';
        quickRisk.innerText = quickRiskVal;
        quickRisk.className = `q-value ${quickRiskVal === 'LOW' ? 'text-green' : (quickRiskVal === 'HIGH' ? 'text-red' : 'text-orange')}`;

        // 3. Targets Widget — prefer verified tech_ctx for CMP, synthesis for targets
        let cmpVal = parseCtxMetric(tech_ctx, 'Current Price', /Current Price\s*:\s*([₹$]?[\d,]+\.?\d*)/i);
        if (!cmpVal || cmpVal === 'N/A') cmpVal = extractPrice(synthesis, "Current Market Price");
        document.getElementById('tgt-cmp').innerText = cmpVal || 'N/A';

        // Buy Zone range
        const buyZoneMatch = synthesis.match(/Suggested Buy Zone[^₹$\d]*([₹$]?\s*\d+[\d,]*\.?\d*(?:\s*[–\-]\s*[₹$]?\s*\d+[\d,]*\.?\d*)?)/i);
        const buyZoneVal = buyZoneMatch ? buyZoneMatch[1].trim() : 'N/A';
        document.getElementById('tgt-buy-zone').innerText = buyZoneVal;

        document.getElementById('tgt-stop-loss').innerText = extractPrice(synthesis, "Stop Loss");
        document.getElementById('tgt-st-target').innerText = extractPrice(synthesis, "Short-term Profit Target", "Short-term Target");
        document.getElementById('tgt-1y-target').innerText = extractPrice(synthesis, "1-Year Outlook Target", "1-Year Price Target", "1-Year Target", "1-Year Outlook");

        // 4. Fundamental metrics — use verified fund_ctx for reliable numbers
        const fundScore = extractScore(fundamental, "Fundamental Score");
        const fundScoreEl = document.getElementById('fund-score-val');
        fundScoreEl.innerText = fundScore;
        fundScoreEl.className = `font-bold ${getScoreColorClass(fundScore)}`;

        document.getElementById('fund-pe-val').innerText =
            parseCtxMetric(fund_ctx, 'P/E Trailing', /P\/E Trailing\s*:\s*([\d.]+|N\/A)/i) ||
            parseCtxMetric(fund_ctx, 'P/E Forward', /P\/E Forward\s*:\s*([\d.]+|N\/A)/i) ||
            'N/A';
        document.getElementById('fund-pb-val').innerText =
            parseCtxMetric(fund_ctx, 'Price-to-Book', /Price-to-Book[^:]*:\s*([\d.]+|N\/A)/i) || 'N/A';
        document.getElementById('fund-de-val').innerText =
            parseCtxMetric(fund_ctx, 'Debt-to-Equity', /Debt-to-Equity\s*:\s*([\d.]+|N\/A)/i) ||
            parseCtxMetric(risk_ctx, 'Debt-to-Equity', /Debt-to-Equity\s*:\s*([\d.]+|N\/A)/i) ||
            'N/A';
        document.getElementById('fund-cr-val').innerText =
            parseCtxMetric(fund_ctx, 'Current Ratio', /Current Ratio\s*:\s*([\d.]+|N\/A)/i) ||
            parseCtxMetric(risk_ctx, 'Current Ratio', /Current Ratio\s*:\s*([\d.]+|N\/A)/i) ||
            'N/A';
        document.getElementById('fund-gm-val').innerText =
            parseCtxMetric(fund_ctx, 'Gross Margin', /Gross Margin\s*:\s*([\d.]+%|N\/A)/i) || 'N/A';
        document.getElementById('fund-nm-val').innerText =
            parseCtxMetric(fund_ctx, 'Net Profit Margin', /Net Profit Margin\s*:\s*([\d.]+%|N\/A)/i) || 'N/A';

        // 5. Technical metrics — use verified tech_ctx for RSI
        const techScore = extractScore(technical, "Technical Score");
        const techScoreEl = document.getElementById('tech-score-val');
        techScoreEl.innerText = techScore;
        techScoreEl.className = `g-val font-bold ${getScoreColorClass(techScore)}`;

        let techScoreNum = parseFloat(techScore) || 5;
        document.getElementById('tech-score-bar').style.width = `${techScoreNum * 10}%`;
        document.getElementById('tech-score-bar').className = `bar-fill ${getScoreBgClass(techScore)}`;

        // RSI from verified tech_ctx
        const rsiRaw = parseCtxMetric(tech_ctx, 'RSI (14)', /RSI\s*\(14\)\s*:\s*([\d.]+)/i);
        const rsiVal = rsiRaw || 'N/A';
        const rsiNum = parseFloat(rsiVal) || 50;
        document.getElementById('tech-rsi-val').innerText = rsiVal;
        const rsiBar = document.getElementById('tech-rsi-bar');
        rsiBar.style.width = `${Math.min(rsiNum, 100)}%`;

        const rsiStatusEl = document.getElementById('tech-rsi-status');
        if (rsiNum > 70) {
            rsiStatusEl.innerText = 'Overbought (Bearish)';
            rsiBar.className = 'bar-fill bg-red';
        } else if (rsiNum < 30) {
            rsiStatusEl.innerText = 'Oversold (Bullish)';
            rsiBar.className = 'bar-fill bg-green';
        } else {
            rsiStatusEl.innerText = 'Neutral';
            rsiBar.className = 'bar-fill bg-blue';
        }

        const trendMatch = technical.match(/Overall Trend[:\s]*([^\n.]+)/i) || technical.match(/trend[^:\n]*[:\s]*([^\n.]+)/i);
        document.getElementById('tech-trend-status').innerText = trendMatch ? trendMatch[1].trim() : 'Neutral Trend';

        // 6. Risk metrics — all from verified risk_ctx
        const riskComposite = risk_ctx.match(/Composite Risk Score\s*:\s*([\d.]+)\/10/i);
        const compositeVal = riskComposite ? `${riskComposite[1]}/10` : 'N/A';
        const riskCompositeEl = document.getElementById('risk-composite-val');
        riskCompositeEl.innerText = compositeVal;
        riskCompositeEl.className = `font-bold ${getScoreColorClass(compositeVal)}`;

        const riskLevelMatch = risk_ctx.match(/Overall Risk Level\s*:\s*(\w+)/i);
        const riskLevelVal = riskLevelMatch ? riskLevelMatch[1] : 'N/A';
        const riskLevelEl = document.getElementById('risk-level-val');
        riskLevelEl.innerText = riskLevelVal;
        riskLevelEl.className = `font-bold ${riskLevelVal === 'Low' ? 'text-green' : (riskLevelVal === 'High' ? 'text-red' : 'text-orange')}`;

        document.getElementById('risk-beta-val').innerText =
            parseCtxMetric(risk_ctx, 'Beta', /Beta\s*:\s*([\d.]+|N\/A)/i) || 'N/A';
        document.getElementById('risk-vol-val').innerText =
            parseCtxMetric(risk_ctx, 'Annualized Volatility', /Annualized Volatility\s*:\s*([\d.]+%|N\/A)/i) || 'N/A';
        document.getElementById('risk-mdd-val').innerText =
            parseCtxMetric(risk_ctx, 'Max Drawdown', /Max Drawdown[^:]*:\s*(-?[\d.]+%|N\/A)/i) || 'N/A';
        document.getElementById('risk-sharpe-val').innerText =
            parseCtxMetric(risk_ctx, 'Sharpe Ratio', /Sharpe Ratio\s*:\s*(-?[\d.]+|N\/A)/i) || 'N/A';

        // 7. Sentiment Metrics and Donut Chart
        const sentScore = extractScore(sentiment, "Sentiment Score");
        const sentScoreEl = document.getElementById('sent-score-val');
        sentScoreEl.innerText = sentScore;
        sentScoreEl.className = `font-bold ${getScoreColorClass(sentScore)}`;

        // Extract "Overall Sentiment Assessment" verdict — the LLM writes it as a section
        // header (## Overall Sentiment Assessment) with the verdict on the next line.
        let sentVerdictVal = 'NEUTRAL';
        const _VERDICT_KW = /\b(STRONGLY\s+BULLISH|STRONGLY\s+BEARISH|VERY\s+BULLISH|VERY\s+BEARISH|MILDLY\s+BULLISH|MILDLY\s+BEARISH|BULLISH|BEARISH|NEUTRAL)\b/i;
        // Strategy 1: section header — scrape the body below "## Overall Sentiment Assessment"
        const _sectionM = sentiment.match(/##\s*Overall\s+Sentiment\s+Assessment([\s\S]*?)(?=\n##|$)/i);
        if (_sectionM) {
            const _kw = _sectionM[1].match(_VERDICT_KW);
            if (_kw) sentVerdictVal = _kw[1].trim();
        }
        // Strategy 2: inline colon format — "Overall Sentiment Assessment: BULLISH"
        if (sentVerdictVal === 'NEUTRAL') {
            const _inline = sentiment.match(/Overall\s+Sentiment\b[^:\n]*:\s*([^\n.]+)/i) ||
                            sentiment.match(/Overall\s+(?:Market\s+)?Verdict\b[^:\n]*:\s*([^\n.]+)/i);
            if (_inline) sentVerdictVal = _inline[1].trim();
        }
        // Strategy 3: fallback — first BULLISH/BEARISH keyword in synthesis
        if (sentVerdictVal === 'NEUTRAL') {
            const _synKw = synthesis.match(_VERDICT_KW);
            if (_synKw) sentVerdictVal = _synKw[1].trim();
        }
        document.getElementById('sent-verdict-val').innerText = sentVerdictVal;
        document.getElementById('sent-verdict-val').className = `font-bold ${getSentimentColorClass(sentVerdictVal)}`;

        // Sentiment Donut Chart
        const scoreNum = parseFloat(sentScore) || 5.0;
        const bullishPct = Math.round(scoreNum * 10);
        const bearishPct = 100 - bullishPct;
        renderSentimentChart(bullishPct, bearishPct);
    }

    // Dynamic Chart rendering
    function renderSentimentChart(bullish, bearish) {
        const ctx = document.getElementById('sentimentChartCanvas').getContext('2d');
        
        if (sentimentChart) {
            sentimentChart.destroy();
        }

        sentimentChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Bullish', 'Bearish'],
                datasets: [{
                    data: [bullish, bearish],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: '#0f1319',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#9ca3af',
                            font: { family: 'Inter', size: 11 }
                        }
                    }
                }
            }
        });
    }

    // Helper functions for parsing

    // Parse a value from a structured verified context block using a specific regex.
    // Falls back to 'N/A' if not found or if the captured value is 'N/A'.
    function parseCtxMetric(ctx, _label, pattern) {
        if (!ctx) return null;
        const match = ctx.match(pattern);
        if (!match) return null;
        const val = match[1].trim();
        return (val && val !== 'N/A') ? val : null;
    }

    function extractPrice(text, ...labels) {
        for (const label of labels) {
            // Accept ₹ or $ as optional currency prefix before the digits
            const pattern = new RegExp(`${label}[^₹$0-9]*([₹$]?\\s*\\d+[\\d,]*\\.?\\d*)`, 'i');
            const match = text.match(pattern);
            if (match) return match[1].replace(/\s+/g, '');
        }
        return 'N/A';
    }

    function extractScore(text, label) {
        const pattern = new RegExp(`${label}[^\\d]*(\\d+(?:\\.\\d+)?)[/\\\\]10`, 'i');
        const match = text.match(pattern);
        return match ? `${match[1]}/10` : 'N/A';
    }

    function extractLabelValue(text, label) {
        const pattern = new RegExp(`${label}[^:\\n]*:\\s*([^\\n(]+)`, 'i');
        const match = text.match(pattern);
        return match ? match[1].trim() : 'N/A';
    }

    function extractTableValue(text, keywords) {
        for (const kw of keywords) {
            const escapedKw = kw.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
            const pattern = new RegExp(`${escapedKw}[^:\\n|]*[:|]\\s*(-?\\d+,?\\.?\\d*%?)`, 'i');
            const match = text.match(pattern);
            if (match) return match[1].trim();
        }
        return 'N/A';
    }

    function getScoreColorClass(scoreStr) {
        if (!scoreStr || scoreStr === 'N/A') return 'text-white';
        const num = parseFloat(scoreStr.split('/')[0]);
        if (isNaN(num)) return 'text-white';
        if (num >= 7) return 'text-green';
        if (num >= 4) return 'text-orange';
        return 'text-red';
    }

    function getScoreBgClass(scoreStr) {
        if (!scoreStr || scoreStr === 'N/A') return '';
        const num = parseFloat(scoreStr.split('/')[0]);
        if (isNaN(num)) return '';
        if (num >= 7) return 'bg-green';
        if (num >= 4) return 'bg-orange';
        return 'bg-red';
    }

    function getSentimentColorClass(verdict) {
        const v = verdict.toUpperCase();
        if (v.includes('BULLISH')) return 'text-green';
        if (v.includes('BEARISH')) return 'text-red';
        return 'text-orange';
    }

    function showErrorCard(message) {
        const existing = document.getElementById('error-card');
        if (existing) existing.remove();

        const card = document.createElement('div');
        card.id = 'error-card';
        card.style.cssText = [
            'position:fixed', 'bottom:32px', 'left:50%', 'transform:translateX(-50%)',
            'z-index:9999', 'background:#1e1216', 'border:1px solid #c0392b',
            'border-radius:12px', 'padding:16px 20px', 'max-width:520px', 'width:90%',
            'box-shadow:0 8px 32px rgba(0,0,0,0.5)', 'display:flex', 'gap:12px', 'align-items:flex-start'
        ].join(';');

        card.innerHTML = `
            <span style="font-size:20px;flex-shrink:0;">⚠️</span>
            <div style="flex:1">
                <div style="color:#e74c3c;font-weight:600;font-size:14px;margin-bottom:4px">Analysis Failed</div>
                <div style="color:#ccc;font-size:13px;line-height:1.5">${escapeHtml(message)}</div>
            </div>
            <button onclick="document.getElementById('error-card').remove()"
                style="background:none;border:none;color:#888;font-size:18px;cursor:pointer;flex-shrink:0;line-height:1">✕</button>`;

        document.body.appendChild(card);
        setTimeout(() => { if (card.parentNode) card.remove(); }, 12000);
    }

    function resetAppToSearch() {
        if (pollingInterval) clearInterval(pollingInterval);

        searchSection.style.display = 'flex';
        chooserSection.style.display = 'none';
        loadingSection.style.display = 'none';
        dashboardSection.style.display = 'none';
        headerMeta.style.display = 'none';

        stockInput.value = '';
        searchBtn.disabled = false;
        searchBtn.querySelector('span').innerText = 'Analyze Stock';
    }

    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, function(m) { return map[m]; });
    }
});
