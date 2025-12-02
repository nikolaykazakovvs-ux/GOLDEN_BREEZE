import sys
sys.path.insert(0, '.')

try:
    print("Importing direction_model...")
    from aimodule.models.direction_model import DirectionPredictor
    print("✓ DirectionPredictor imported")
    
    print("Importing regime_model...")
    from aimodule.models.regime_model import RegimeClusterModel
    print("✓ RegimeClusterModel imported")
    
    print("Importing sentiment_model...")
    from aimodule.models.sentiment_model import LexiconSentimentModel
    print("✓ LexiconSentimentModel imported")
    
    print("\nCreating instances...")
    d = DirectionPredictor()
    print("✓ DirectionPredictor instance created")
    
    r = RegimeClusterModel()
    print("✓ RegimeClusterModel instance created")
    
    s = LexiconSentimentModel()
    print("✓ LexiconSentimentModel instance created")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
