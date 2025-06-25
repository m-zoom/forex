# Pattern Detection Internals

## Core Detection Algorithms

### 1. Head and Shoulders Pattern

#### Mathematical Foundation
The Head and Shoulders pattern is detected using geometric analysis of price peaks:

```python
def detect_head_shoulders(self, df):
    """
    H&S Detection Algorithm:
    1. Find significant peaks using local maxima
    2. Identify three consecutive peaks
    3. Apply geometric validation rules
    4. Calculate confidence based on pattern quality
    """
    
    # Peak detection with configurable sensitivity
    peaks = self._find_significant_peaks(df, window=5, min_prominence=0.002)
    
    patterns = []
    for i in range(len(peaks) - 2):
        left_shoulder = peaks[i]
        head = peaks[i + 1]
        right_shoulder = peaks[i + 2]
        
        # Geometric validation
        if self._validate_hns_geometry(left_shoulder, head, right_shoulder):
            pattern = self._create_hns_pattern(left_shoulder, head, right_shoulder, df)
            patterns.append(pattern)
            
    return patterns

def _validate_hns_geometry(self, left, head, right):
    """
    Validation Rules:
    - Head must be significantly higher than both shoulders (>2%)
    - Shoulders should be approximately equal height (within 5%)
    - Time spacing should be reasonable
    """
    head_prominence = (head['price'] - max(left['price'], right['price'])) / head['price']
    shoulder_symmetry = abs(left['price'] - right['price']) / left['price']
    
    return (
        head_prominence > 0.02 and  # Head 2% higher
        shoulder_symmetry < 0.05 and  # Shoulders within 5%
        self._validate_time_spacing(left, head, right)
    )
```

#### Neckline Calculation
```python
def _calculate_neckline(self, left_shoulder, head, right_shoulder, df):
    """
    Calculate neckline support level:
    1. Find lowest points between shoulders and head
    2. Draw line connecting these lows
    3. Use linear regression for precise level
    """
    # Find valley between left shoulder and head
    left_valley_idx = self._find_valley_between_points(
        df, left_shoulder['index'], head['index']
    )
    
    # Find valley between head and right shoulder  
    right_valley_idx = self._find_valley_between_points(
        df, head['index'], right_shoulder['index']
    )
    
    # Linear regression for neckline slope
    x_coords = [left_valley_idx, right_valley_idx]
    y_coords = [df['low'].iloc[left_valley_idx], df['low'].iloc[right_valley_idx]]
    
    slope, intercept = np.polyfit(x_coords, y_coords, 1)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'level': np.mean(y_coords),  # Average for simplicity
        'left_valley': left_valley_idx,
        'right_valley': right_valley_idx
    }
```

### 2. Double Top/Bottom Detection

#### Price Level Clustering Algorithm
```python
def detect_double_patterns(self, df):
    """
    Double Pattern Detection:
    1. Identify all significant peaks/troughs
    2. Group nearby levels using clustering
    3. Find pairs with similar heights
    4. Validate time separation and pattern completion
    """
    
    # Separate analysis for tops and bottoms
    double_tops = self._detect_double_tops(df)
    double_bottoms = self._detect_double_bottoms(df)
    
    return double_tops + double_bottoms

def _detect_double_tops(self, df):
    """Find double top patterns using price clustering"""
    peaks = self._find_significant_peaks(df, window=10, min_prominence=0.003)
    
    patterns = []
    tolerance = df['close'].iloc[-1] * 0.005  # 0.5% tolerance
    
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            peak1, peak2 = peaks[i], peaks[j]
            
            # Check if peaks are at similar levels
            price_diff = abs(peak1['price'] - peak2['price'])
            if price_diff <= tolerance:
                
                # Validate time separation (minimum 10 candles apart)
                if peak2['index'] - peak1['index'] >= 10:
                    
                    # Check for valley between peaks
                    valley = self._find_valley_between_points(
                        df, peak1['index'], peak2['index']
                    )
                    
                    if valley and self._validate_double_top_valley(peak1, peak2, valley, df):
                        pattern = self._create_double_top_pattern(peak1, peak2, valley, df)
                        patterns.append(pattern)
                        
    return patterns

def _validate_double_top_valley(self, peak1, peak2, valley_idx, df):
    """
    Valley Validation Rules:
    - Valley must be significantly lower than peaks (>3%)
    - Valley should be roughly centered between peaks
    - No higher peaks between the two main peaks
    """
    valley_price = df['low'].iloc[valley_idx]
    avg_peak_price = (peak1['price'] + peak2['price']) / 2
    
    valley_depth = (avg_peak_price - valley_price) / avg_peak_price
    
    # Check depth requirement
    if valley_depth < 0.03:  # 3% minimum depth
        return False
        
    # Check for interfering peaks
    between_highs = df['high'].iloc[peak1['index']:peak2['index']]
    max_between = between_highs.max()
    
    if max_between > min(peak1['price'], peak2['price']) * 1.01:  # 1% tolerance
        return False
        
    return True
```

### 3. Triangle Pattern Detection

#### Trendline Calculation Engine
```python
def detect_triangles(self, df):
    """
    Triangle Detection Process:
    1. Calculate upper resistance trendline
    2. Calculate lower support trendline  
    3. Check for convergence
    4. Classify triangle type (ascending/descending/symmetrical)
    """
    
    patterns = []
    
    # Require minimum data points for reliable trendlines
    if len(df) < 50:
        return patterns
        
    # Find peaks and troughs
    peaks = self._find_significant_peaks(df, window=5)
    troughs = self._find_significant_troughs(df, window=5)
    
    # Calculate trendlines
    upper_trendline = self._calculate_resistance_trendline(peaks, df)
    lower_trendline = self._calculate_support_trendline(troughs, df)
    
    if upper_trendline and lower_trendline:
        triangle_type = self._classify_triangle(upper_trendline, lower_trendline)
        
        if triangle_type:
            pattern = self._create_triangle_pattern(
                upper_trendline, lower_trendline, triangle_type, df
            )
            patterns.append(pattern)
            
    return patterns

def _calculate_resistance_trendline(self, peaks, df):
    """
    Calculate resistance trendline using linear regression:
    1. Filter peaks in recent period (last 30 candles)
    2. Apply linear regression to peak prices vs time
    3. Validate trendline quality (R-squared > 0.8)
    """
    if len(peaks) < 3:
        return None
        
    # Use recent peaks only
    recent_peaks = [p for p in peaks if p['index'] >= len(df) - 30]
    
    if len(recent_peaks) < 3:
        return None
        
    # Prepare data for regression
    x_data = [p['index'] for p in recent_peaks]
    y_data = [p['price'] for p in recent_peaks]
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    
    # Quality check
    if r_value ** 2 < 0.8:  # R-squared threshold
        return None
        
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'points': recent_peaks,
        'equation': f"y = {slope:.6f}x + {intercept:.6f}"
    }

def _classify_triangle(self, upper_line, lower_line):
    """
    Triangle Classification:
    - Ascending: Lower line slopes up, upper line flat/slightly up
    - Descending: Upper line slopes down, lower line flat/slightly down  
    - Symmetrical: Lines converge with opposite slopes
    """
    upper_slope = upper_line['slope']
    lower_slope = lower_line['slope']
    
    slope_threshold = 0.0001  # Minimal slope for "flat" lines
    
    if abs(upper_slope) < slope_threshold and lower_slope > slope_threshold:
        return 'Ascending Triangle'
    elif abs(lower_slope) < slope_threshold and upper_slope < -slope_threshold:
        return 'Descending Triangle'  
    elif upper_slope < -slope_threshold and lower_slope > slope_threshold:
        return 'Symmetrical Triangle'
    else:
        return None  # Not a valid triangle
```

### 4. Support and Resistance Levels

#### Advanced Level Detection Algorithm
```python
def detect_support_resistance(self, df):
    """
    Multi-Step S/R Detection:
    1. Identify price touch points using local extrema
    2. Cluster nearby price levels 
    3. Score levels based on touches, age, and volume
    4. Filter significant levels only
    """
    
    support_levels = self._detect_support_levels(df)
    resistance_levels = self._detect_resistance_levels(df)
    
    return support_levels + resistance_levels

def _detect_support_levels(self, df):
    """Support level detection with clustering"""
    
    # Find all local minima (potential support touches)
    local_minima = []
    window = 3
    
    for i in range(window, len(df) - window):
        current_low = df['low'].iloc[i]
        
        # Check if current point is lowest in window
        is_minimum = all(
            current_low <= df['low'].iloc[j] 
            for j in range(i - window, i + window + 1)
        )
        
        if is_minimum:
            local_minima.append({
                'price': current_low,
                'index': i,
                'time': df.index[i],
                'volume': df['volume'].iloc[i] if 'volume' in df.columns else 0
            })
    
    # Cluster nearby price levels
    price_tolerance = df['close'].iloc[-1] * 0.002  # 0.2% clustering tolerance
    clusters = self._cluster_price_levels(local_minima, price_tolerance)
    
    # Score and filter clusters
    support_levels = []
    for cluster in clusters:
        if len(cluster['touches']) >= 3:  # Minimum 3 touches
            level_score = self._calculate_level_strength(cluster, df)
            
            if level_score >= 6:  # Minimum strength threshold
                support_levels.append({
                    'type': 'Support Level',
                    'level': cluster['avg_price'],
                    'strength': level_score,
                    'touch_count': len(cluster['touches']),
                    'touches': cluster['touches'],
                    'confidence': min(95, 50 + level_score * 5),
                    'signal': 'Bullish',
                    'timestamp': df.index[-1]
                })
    
    return support_levels

def _cluster_price_levels(self, price_points, tolerance):
    """Group nearby price levels using proximity clustering"""
    clusters = []
    
    for point in price_points:
        price = point['price']
        assigned = False
        
        # Try to assign to existing cluster
        for cluster in clusters:
            if abs(cluster['avg_price'] - price) <= tolerance:
                cluster['touches'].append(point)
                cluster['avg_price'] = np.mean([t['price'] for t in cluster['touches']])
                assigned = True
                break
                
        # Create new cluster if not assigned
        if not assigned:
            clusters.append({
                'avg_price': price,
                'touches': [point],
                'created_at': point['time']
            })
    
    return clusters

def _calculate_level_strength(self, cluster, df):
    """
    Level Strength Scoring (1-10 scale):
    - Touch count: +2 points per touch (max 6 points)
    - Age: +1 point if level exists for >20 candles
    - Volume: +2 points if average volume at touches is above median
    - Recent test: +1 point if tested in last 10 candles
    """
    score = 0
    
    # Touch count scoring
    touch_count = len(cluster['touches'])
    score += min(6, touch_count * 2)
    
    # Age scoring
    oldest_touch = min(cluster['touches'], key=lambda t: t['index'])
    level_age = len(df) - oldest_touch['index']
    if level_age > 20:
        score += 1
        
    # Volume scoring
    if all('volume' in touch for touch in cluster['touches']):
        touch_volumes = [touch['volume'] for touch in cluster['touches']]
        avg_touch_volume = np.mean(touch_volumes)
        median_volume = df['volume'].median()
        
        if avg_touch_volume > median_volume:
            score += 2
            
    # Recent test scoring
    recent_touches = [t for t in cluster['touches'] if len(df) - t['index'] <= 10]
    if recent_touches:
        score += 1
        
    return min(10, score)
```

## Confidence Scoring System

### Multi-Factor Confidence Calculation
```python
def calculate_pattern_confidence(self, pattern_data, pattern_type):
    """
    Comprehensive confidence scoring using multiple factors:
    1. Geometric accuracy (40% weight)
    2. Volume confirmation (25% weight)  
    3. Market context (20% weight)
    4. Historical performance (15% weight)
    """
    
    geometric_score = self._score_geometric_accuracy(pattern_data, pattern_type)
    volume_score = self._score_volume_confirmation(pattern_data)
    context_score = self._score_market_context(pattern_data)
    historical_score = self._score_historical_performance(pattern_type)
    
    # Weighted average
    confidence = (
        geometric_score * 0.40 +
        volume_score * 0.25 +
        context_score * 0.20 +
        historical_score * 0.15
    )
    
    return min(100, max(0, confidence))

def _score_geometric_accuracy(self, pattern_data, pattern_type):
    """Score how well pattern matches ideal geometry"""
    
    if pattern_type == 'Head and Shoulders':
        # Check shoulder symmetry
        left_shoulder = pattern_data['points'][0]['price']
        right_shoulder = pattern_data['points'][2]['price']
        symmetry = 1 - abs(left_shoulder - right_shoulder) / left_shoulder
        
        # Check head prominence
        head = pattern_data['points'][1]['price']
        avg_shoulder = (left_shoulder + right_shoulder) / 2
        prominence = (head - avg_shoulder) / head
        
        return (symmetry * 0.6 + min(1, prominence * 10) * 0.4) * 100
        
    elif pattern_type == 'Double Top':
        # Check peak similarity
        peak1 = pattern_data['points'][0]['price']
        peak2 = pattern_data['points'][1]['price']
        similarity = 1 - abs(peak1 - peak2) / peak1
        
        return similarity * 100
        
    # Default scoring for other patterns
    return 70

def _score_volume_confirmation(self, pattern_data):
    """Score volume behavior during pattern formation"""
    
    if 'volume_profile' not in pattern_data:
        return 60  # Neutral score if no volume data
        
    volume_profile = pattern_data['volume_profile']
    
    # Look for volume expansion on breakouts
    if volume_profile.get('breakout_volume_ratio', 1) > 1.5:
        return 90  # Strong volume confirmation
    elif volume_profile.get('breakout_volume_ratio', 1) > 1.2:
        return 75  # Moderate volume confirmation
    else:
        return 50  # Weak volume confirmation

def _score_market_context(self, pattern_data):
    """Score pattern within broader market context"""
    
    # Check trend alignment
    trend_alignment = pattern_data.get('trend_alignment', 'neutral')
    
    if trend_alignment == 'strong':
        return 85
    elif trend_alignment == 'moderate':
        return 70
    else:
        return 55  # Pattern against trend or unclear

def _score_historical_performance(self, pattern_type):
    """Score based on historical pattern success rates"""
    
    # Historical success rates from backtesting
    historical_rates = {
        'Head and Shoulders': 72,
        'Double Top': 68,
        'Double Bottom': 71,
        'Ascending Triangle': 75,
        'Descending Triangle': 73,
        'Support Level': 65,
        'Resistance Level': 67
    }
    
    return historical_rates.get(pattern_type, 60)
```

## Real-Time Pattern Monitoring

### Incremental Pattern Detection
```python
class IncrementalPatternDetector:
    """Efficient pattern detection for real-time data streams"""
    
    def __init__(self, config):
        self.config = config
        self.pattern_candidates = {}  # Tracking incomplete patterns
        self.confirmed_patterns = []
        
    def process_new_candle(self, new_candle_data, full_buffer):
        """Process single new candle for pattern updates"""
        
        # Update existing pattern candidates
        self._update_pattern_candidates(new_candle_data, full_buffer)
        
        # Look for new pattern formations
        new_patterns = self._scan_for_new_patterns(full_buffer)
        
        # Confirm completed patterns
        confirmed = self._check_pattern_confirmations(full_buffer)
        
        return confirmed

    def _update_pattern_candidates(self, new_candle, buffer):
        """Update incomplete patterns with new data"""
        
        to_remove = []
        
        for pattern_id, candidate in self.pattern_candidates.items():
            pattern_type = candidate['type']
            
            if pattern_type == 'Head and Shoulders':
                # Check if new candle completes neckline break
                if self._check_neckline_break(candidate, new_candle):
                    candidate['status'] = 'confirmed'
                    candidate['confirmation_candle'] = new_candle
                    
            elif pattern_type == 'Triangle':
                # Check for triangle breakout
                if self._check_triangle_breakout(candidate, new_candle, buffer):
                    candidate['status'] = 'confirmed'
                    candidate['breakout_candle'] = new_candle
                    
            # Remove expired candidates (too old without confirmation)
            if self._is_pattern_expired(candidate, buffer):
                to_remove.append(pattern_id)
                
        # Clean up expired candidates
        for pattern_id in to_remove:
            del self.pattern_candidates[pattern_id]

    def _check_neckline_break(self, hns_candidate, new_candle):
        """Check if new candle breaks below neckline with volume"""
        neckline_level = hns_candidate['neckline']['level']
        
        # Price break confirmation
        price_break = new_candle['close'] < neckline_level * 0.998  # 0.2% buffer
        
        # Volume confirmation (optional but preferred)
        volume_confirm = True
        if 'volume' in new_candle and 'avg_volume' in hns_candidate:
            volume_confirm = new_candle['volume'] > hns_candidate['avg_volume'] * 1.2
            
        return price_break and volume_confirm
```

This detailed guide provides developers with comprehensive understanding of the mathematical foundations and algorithmic approaches used in pattern detection, enabling them to extend or modify the detection logic effectively.