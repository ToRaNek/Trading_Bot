"""Script de test pour le système de trading en temps réel"""

from trading import Portfolio
from datetime import datetime

def test_portfolio():
    """Test le système de portefeuille"""
    print("\n=== TEST DU PORTEFEUILLE ===\n")

    # Créer un portfolio de test
    portfolio = Portfolio(initial_cash=1000.0, save_file='test_portfolio.json')
    print(f"Capital initial: ${portfolio.cash:.2f}")

    # Test d'achat
    print("\n--- Test d'achat ---")
    success = portfolio.buy('AAPL', 150.0, 5, datetime.now())
    print(f"Achat réussi: {success}")
    print(f"Cash restant: ${portfolio.cash:.2f}")
    print(f"Positions: {portfolio.positions}")

    # Test de vente
    print("\n--- Test de vente ---")
    success = portfolio.sell('AAPL', 160.0, timestamp=datetime.now())
    print(f"Vente réussie: {success}")
    print(f"Cash après vente: ${portfolio.cash:.2f}")
    print(f"Positions: {portfolio.positions}")

    # Calcul de performance
    print("\n--- Performance ---")
    current_prices = {'AAPL': 160.0}
    performance = portfolio.get_performance(current_prices)

    print(f"Valeur totale: ${performance['total_value']:.2f}")
    print(f"Retour: ${performance['total_return']:.2f} ({performance['total_return_pct']:+.2f}%)")
    print(f"Trades: {performance['total_trades']}")
    print(f"Win rate: {performance['win_rate']:.1f}%")

    # Réinitialiser le portfolio
    portfolio.reset()
    print("\n--- Portfolio réinitialisé ---")
    print(f"Cash: ${portfolio.cash:.2f}")
    print(f"Positions: {portfolio.positions}")

    print("\n=== TESTS TERMINÉS ===\n")

if __name__ == "__main__":
    test_portfolio()
