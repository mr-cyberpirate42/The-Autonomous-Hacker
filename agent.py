#!/usr/bin/env python3
"""
Autonomous Hacker Agent - ADK Compatible
Built for Kaggle Competitions
"""

import pandas as pd
import numpy as np
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VibeDetector:
    """Detects competition patterns and characteristics"""
    
    def __init__(self):
        self.meta_features = {}
        
    def analyze_dataset(self, data):
        """Analyze dataset and extract meta-features"""
        logger.info("🔍 Detecting competition vibe...")
        
        # Basic meta-features
        self.meta_features = {
            'samples': len(data),
            'features': len(data.columns) - 1,  # exclude target
            'numeric_count': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_count': len(data.select_dtypes(include=['object']).columns),
            'missing_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns)),
            'target_type': self._detect_target_type(data)
        }
        
        vibe = self._classify_vibe()
        return vibe
    
    def _detect_target_type(self, data):
        """Detect if classification or regression"""
        # Check for common target column names
        target_cols = ['target', 'y', 'label', 'output', 'class']
        target_col = None
        
        for col in target_cols:
            if col in data.columns:
                target_col = col
                break
        
        # Use last column if no standard name found
        if target_col is None and len(data.columns) > 0:
            target_col = data.columns[-1]
        
        if target_col is None:
            return 'unknown'
        
        try:
            unique_vals = data[target_col].nunique()
            # Classification if discrete values, regression if continuous
            ratio = unique_vals / len(data)
            if unique_vals < 20 or ratio < 0.05:
                return 'classification'
            return 'regression'
        except Exception:
            return 'unknown'
    
    def _classify_vibe(self):
        """Classify competition based on meta-features"""
        vibe_types = []
        
        if self.meta_features['features'] > 100:
            vibe_types.append("high_dimensional")
        if self.meta_features['missing_ratio'] > 0.3:
            vibe_types.append("messy_data")
        if self.meta_features['numeric_count'] / self.meta_features['features'] > 0.8:
            vibe_types.append("numeric_dominant")
        if self.meta_features['categorical_count'] / self.meta_features['features'] > 0.5:
            vibe_types.append("categorical_heavy")
            
        return {
            'vibe_types': vibe_types,
            'meta_features': self.meta_features,
            'recommended_models': self._get_recommended_models(vibe_types)
        }
    
    def _get_recommended_models(self, vibe_types):
        """Get model recommendations based on vibe"""
        model_map = {
            'high_dimensional': ['LightGBM', 'XGBoost'],
            'messy_data': ['RandomForest', 'XGBoost'],
            'numeric_dominant': ['LightGBM', 'XGBoost', 'NeuralNetwork'],
            'categorical_heavy': ['CatBoost', 'LightGBM']
        }
        
        models = []
        for vibe in vibe_types:
            models.extend(model_map.get(vibe, []))
        
        return list(set(models))

class StrategyGenerator:
    """Generates competition strategies based on vibe"""
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge()
        
    def _load_knowledge(self):
        """Load historical competition patterns"""
        # Try multiple possible filenames with proper path resolution
        base_dir = Path(__file__).parent
        possible_paths = [
            base_dir / 'knowledge' / 'competition_patterns.json',
            base_dir / 'knowledge' / 'competition_pattern.json',
            base_dir / 'knowledge' / 'patterns.json'
        ]
        
        for knowledge_file in possible_paths:
            try:
                if knowledge_file.exists():
                    with open(knowledge_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        logger.info(f"✅ Loaded knowledge from {knowledge_file}")
                        return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"⚠️ Error loading {knowledge_file}: {e}")
                continue
        
        logger.info("ℹ️ No knowledge base found, starting with empty base")
        return {"patterns": [], "strategies": [], "learned_vibes": []}
    
    def generate_strategy(self, vibe_analysis):
        """Generate competition strategy"""
        logger.info("🎯 Generating winning strategy...")
        
        strategy = {
            'feature_engineering': self._get_feature_plan(vibe_analysis),
            'model_pipeline': self._get_model_pipeline(vibe_analysis),
            'validation_strategy': self._get_validation_plan(vibe_analysis),
            'ensemble_approach': self._get_ensemble_plan(vibe_analysis)
        }
        
        return strategy
    
    def _get_feature_plan(self, vibe):
        """Get feature engineering plan"""
        plans = {
            'high_dimensional': ['variance_threshold', 'correlation_filter'],
            'messy_data': ['robust_imputation', 'outlier_detection'],
            'categorical_heavy': ['target_encoding', 'frequency_encoding']
        }
        
        feature_plan = ['standard_scaling', 'missing_imputation']
        for vibe_type in vibe['vibe_types']:
            feature_plan.extend(plans.get(vibe_type, []))
            
        return list(set(feature_plan))
    
    def _get_model_pipeline(self, vibe):
        """Get model pipeline"""
        base_models = ['LightGBM', 'XGBoost']
        
        # Add vibe-specific models
        if 'high_dimensional' in vibe['vibe_types']:
            base_models.extend(['LinearModel', 'SVM'])
        if 'categorical_heavy' in vibe['vibe_types']:
            base_models.append('CatBoost')
            
        return base_models
    
    def _get_validation_plan(self, vibe):
        """Get validation strategy"""
        if vibe['meta_features']['samples'] < 1000:
            return 'repeated_kfold'
        elif 'time_series' in vibe['vibe_types']:
            return 'time_series_split'
        else:
            return 'stratified_kfold'
    
    def _get_ensemble_plan(self, vibe):
        """Get ensemble strategy"""
        if len(vibe['recommended_models']) >= 3:
            return 'stacking'
        else:
            return 'weighted_average'

class AutonomousHackerAgent:
    """Main Autonomous Hacker Agent - ADK Compatible"""
    
    def __init__(self):
        self.vibe_detector = VibeDetector()
        self.strategy_generator = StrategyGenerator()
        self.competition_history = []
        
    async def process_competition(self, data_path):
        """Main method called by ADK"""
        logger.info("🚀 Autonomous Hacker Agent Activated!")
        logger.info("=" * 50)
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            logger.info(f"📊 Data loaded: {data.shape}")
            
            # Step 1: Vibe Detection
            vibe_analysis = self.vibe_detector.analyze_dataset(data)
            logger.info(f"✅ Vibe detected: {vibe_analysis['vibe_types']}")
            
            # Step 2: Strategy Generation
            strategy = self.strategy_generator.generate_strategy(vibe_analysis)
            logger.info(f"🎯 Strategy generated: {len(strategy['model_pipeline'])} models selected")
            
            # Store results
            result = {
                'timestamp': datetime.now().isoformat(),
                'data_shape': data.shape,
                'vibe_analysis': vibe_analysis,
                'strategy': strategy,
                'status': 'completed'
            }
            
            self.competition_history.append(result)
            
            logger.info("=" * 50)
            logger.info("✅ Analysis Complete!")
            
            return result
        except FileNotFoundError as e:
            logger.error(f"❌ Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error during analysis: {e}", exc_info=True)
            raise

# ADK-Compatible Interface
async def main():
    """ADK entry point"""
    agent = AutonomousHackerAgent()
    
    # For demo purposes - replace with actual data path
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randint(0, 10, 1000),
        'target': np.random.randint(0, 2, 1000)
    })
    sample_data.to_csv('sample_competition.csv', index=False)
    
    results = await agent.process_competition('sample_competition.csv')
    return results

if __name__ == "__main__":
    asyncio.run(main())