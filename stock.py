import pandas as pd
from typing import List, Dict, Tuple
from pulp import *
import argparse
from collections import defaultdict

class CuttingStockSolver:
    def __init__(self, prices_file: str, parts_file: str):
        """Initialize solver with stock lengths and required cuts from CSV files"""
        # Read prices data
        self.prices_df = pd.read_csv(prices_file)
        self.stock_lengths = [
            {
                'length': row['length'] * 12,  # Convert to inches
                'price': row['price']
            }
            for _, row in self.prices_df.iterrows()
        ]
        
        # Read parts data
        self.parts_df = pd.read_csv(parts_file)
        self.required_cuts = [
            {
                'length': row['LEN'],
                'quantity': row['QTY']
            }
            for _, row in self.parts_df.iterrows()
        ]
        
        self.patterns = []
        self.pattern_costs = []
        
        # Calculate theoretical minimums
        self.theoretical_minimums = self.calculate_theoretical_minimums()
        
        # Configure solver
        self.solver = COIN_CMD(path=None, keepFiles=False, msg=False, 
                             options=['AllowableGap', '0.0',
                                    'RatioGap', '0.0',
                                    'MaxNodes', '10000',
                                    'MaxSolutions', '1000',
                                    'TimeLimit', '120'])

    def calculate_theoretical_minimums(self) -> Dict:
        """Calculate theoretical minimums for both material and cost"""
        total_length_needed = sum(cut['length'] * cut['quantity'] 
                                for cut in self.required_cuts)
        
        price_per_inch = [(stock['price'] / stock['length'], stock['length'], stock['price']) 
                         for stock in self.stock_lengths]
        most_efficient = min(price_per_inch, key=lambda x: x[0])
        
        min_boards_by_length = {}
        for stock in self.stock_lengths:
            boards_needed = total_length_needed / stock['length']
            min_boards_by_length[stock['length']] = {
                'boards': boards_needed,
                'cost': boards_needed * stock['price']
            }
        
        min_theoretical_cost = (total_length_needed * most_efficient[0])
        
        all_cuts = []
        for cut in self.required_cuts:
            all_cuts.extend([cut['length']] * cut['quantity'])
        all_cuts.sort(reverse=True)
        
        max_board_length = max(stock['length'] for stock in self.stock_lengths)
        current_board = max_board_length
        total_min_waste = 0
        remaining_cuts = all_cuts.copy()
        
        while remaining_cuts:
            if current_board < remaining_cuts[0]:
                total_min_waste += current_board
                current_board = max_board_length
            cut = remaining_cuts.pop(0)
            current_board -= cut
        
        if current_board > 0:
            total_min_waste += current_board
            
        return {
            'total_length_needed': total_length_needed,
            'min_theoretical_cost': min_theoretical_cost,
            'min_boards_by_length': min_boards_by_length,
            'min_theoretical_waste': total_min_waste,
            'price_efficiency': {
                'board_length': most_efficient[1],
                'price_per_inch': most_efficient[0],
                'total_price': most_efficient[2]
            }
        }

    def generate_initial_patterns(self):
        """Generate initial cutting patterns"""
        # For each stock length, try to fit each type of cut
        for stock in self.stock_lengths:
            for i, cut in enumerate(self.required_cuts):
                # How many pieces of this cut can we get from this stock?
                num_pieces = int(stock['length'] // cut['length'])
                if num_pieces > 0:
                    pattern = [0] * len(self.required_cuts)
                    pattern[i] = num_pieces
                    self.patterns.append(pattern)
                    self.pattern_costs.append(stock['price'])

    def solve_master_problem(self) -> Tuple[List[float], float, List[float]]:
        """Solve the master linear programming problem"""
        prob = LpProblem("Cutting_Stock_4x4", LpMinimize)
        
        pattern_vars = [LpVariable(f"pattern_{i}", 0, None) 
                       for i in range(len(self.patterns))]
        
        prob += lpSum(self.pattern_costs[i] * pattern_vars[i] 
                     for i in range(len(self.patterns)))
        
        demand_constraints = []
        for i in range(len(self.required_cuts)):
            constraint = lpSum(self.patterns[j][i] * pattern_vars[j] 
                             for j in range(len(self.patterns))) >= self.required_cuts[i]['quantity']
            demand_constraints.append(constraint)
            prob += constraint
        
        prob.solve(self.solver)
        
        dual_values = [c.pi for c in demand_constraints]
        return [value(var) for var in pattern_vars], value(prob.objective), dual_values

    def solve_knapsack(self, items: List[Tuple[int, float, float]], 
                      capacity: float, cost: float) -> Tuple[List[int], float]:
        """Solve knapsack subproblem"""
        prob = LpProblem("Pattern_Generation", LpMaximize)
        
        x = [LpVariable(f"cut_{i}", 0, None, LpInteger) for i, _, _ in items]
        prob += lpSum(dual * x[i] for i, _, dual in items)
        prob += lpSum(length * x[i] for i, length, _ in items) <= capacity
        
        prob.solve(self.solver)
        pattern = [int(value(var)) for var in x]
        reduced_cost = cost - sum(dual * value(x[i]) for i, _, dual in items)
        
        return pattern, reduced_cost

    def solve_subproblem(self, dual_values: List[float]) -> Tuple[List[int], float, float]:
        """Find new promising cutting patterns"""
        best_reduced_cost = float('inf')
        best_pattern = None
        best_stock_cost = None
        
        for stock in self.stock_lengths:
            items = [(i, cut['length'], dual_values[i]) 
                    for i, cut in enumerate(self.required_cuts)]
            pattern, reduced_cost = self.solve_knapsack(
                items, stock['length'], stock['price'])
            
            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_pattern = pattern
                best_stock_cost = stock['price']
        
        return best_pattern, best_reduced_cost, best_stock_cost

    def optimize(self) -> Dict:
        """Main optimization routine"""
        self.generate_initial_patterns()
        
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            pattern_usage, objective, dual_values = self.solve_master_problem()
            new_pattern, reduced_cost, stock_cost = self.solve_subproblem(dual_values)
            
            if reduced_cost >= -1e-6:
                break
            
            self.patterns.append(new_pattern)
            self.pattern_costs.append(stock_cost)
            iteration += 1
        
        prob = LpProblem("Cutting_Stock_Final", LpMinimize)
        pattern_vars = [LpVariable(f"pattern_{i}", 0, None, LpInteger) 
                       for i in range(len(self.patterns))]
        
        prob += lpSum(self.pattern_costs[i] * pattern_vars[i] 
                     for i in range(len(self.patterns)))
        
        for i in range(len(self.required_cuts)):
            prob += lpSum(self.patterns[j][i] * pattern_vars[j] 
                         for j in range(len(self.patterns))) >= self.required_cuts[i]['quantity']
        
        prob.solve(self.solver)
        pattern_usage = [value(var) for var in pattern_vars]
        
        result = {
            'total_cost': value(prob.objective),
            'total_waste': self.calculate_waste(pattern_usage),
            'cutting_patterns': self.format_solution(pattern_usage),
            'status': LpStatus[prob.status],
            'iterations': iteration,
            'theoretical_minimums': self.theoretical_minimums,
            'solution_quality': {
                'cost_gap': (value(prob.objective) - self.theoretical_minimums['min_theoretical_cost']) / 
                            self.theoretical_minimums['min_theoretical_cost'] * 100,
                'waste_gap': (self.calculate_waste(pattern_usage) - self.theoretical_minimums['min_theoretical_waste']) /
                            self.theoretical_minimums['min_theoretical_waste'] * 100 if self.theoretical_minimums['min_theoretical_waste'] > 0 else 0
            }
        }
        
        return result

    def calculate_waste(self, pattern_usage: List[float]) -> float:
        """Calculate total waste"""
        total_waste = 0
        for i, usage in enumerate(pattern_usage):
            if usage > 1e-6:
                pattern = self.patterns[i]
                used_length = sum(pattern[j] * self.required_cuts[j]['length'] 
                                for j in range(len(self.required_cuts)))
                stock_length = max(stock['length'] 
                                 for stock in self.stock_lengths 
                                 if abs(stock['price'] - self.pattern_costs[i]) < 1e-6)
                total_waste += (stock_length - used_length) * usage
        return total_waste

    def format_solution(self, pattern_usage: List[float]) -> List[Dict]:
        """Format solution with cut details"""
        cutting_patterns = []
        for i, usage in enumerate(pattern_usage):
            if usage > 1e-6:
                pattern = self.patterns[i]
                cuts = []
                cut_details = []
                for j, count in enumerate(pattern):
                    if count > 0:
                        length = self.required_cuts[j]['length']
                        part_desc = self.parts_df.iloc[j]['LABEL / PART DESCRIPTION']
                        cuts.extend([length] * count)
                        cut_details.append({
                            'length': length,
                            'count': count,
                            'description': part_desc
                        })
                
                stock_length = max(stock['length'] 
                                 for stock in self.stock_lengths 
                                 if abs(stock['price'] - self.pattern_costs[i]) < 1e-6)
                
                cutting_patterns.append({
                    'stock_length': stock_length,
                    'stock_length_feet': stock_length / 12,
                    'cuts': cuts,
                    'cut_details': cut_details,
                    'times_used': int(round(usage))
                })
        return cutting_patterns

def print_simple_solution(result: Dict):
    """Print solution in simple format with just costs and cut lists"""
    print(f"Total Cost: ${result['total_cost']:.2f}")
    print(f"Total Waste: {result['total_waste']:.2f} inches")
    print("Boards Used:")
    for pattern in result['cutting_patterns']:
        for _ in range(pattern['times_used']):
            cuts = [f"{cut:.2f}" for detail in pattern['cut_details'] 
                   for cut in [detail['length']] * detail['count']]
            print(f"Board length: {pattern['stock_length']}\" - Cuts: [{', '.join(cuts)}]")

def print_collapsed_solution(result: Dict):
    """Print solution in collapsed format grouping by board length"""
    print(f"Total Cost: ${result['total_cost']:.2f}")
    print(f"Total Waste: {result['total_waste']:.2f} inches")
    print("Boards Used:")
    
    # Group patterns by board length
    length_patterns = defaultdict(list)
    for pattern in result['cutting_patterns']:
        cuts = [f"{cut:.2f}" for detail in pattern['cut_details'] 
                for cut in [detail['length']] * detail['count']]
        cuts.sort()  # Sort cuts to ensure consistent ordering
        length_patterns[pattern['stock_length']].append({
            'cuts': cuts,
            'quantity': pattern['times_used']
        })
    
    # Print collapsed patterns by board length
    for stock_length in sorted(length_patterns.keys()):
        patterns = length_patterns[stock_length]
        total_quantity = sum(p['quantity'] for p in patterns)
        
        # Group cuts into sublists
        unique_cut_patterns = []
        for pattern in patterns:
            cuts = pattern['cuts']
            if not any(cuts == existing_cuts for existing_cuts in unique_cut_patterns):
                unique_cut_patterns.append(cuts)
        
        # Format the cuts list
        cut_patterns_str = ', '.join(f"[{', '.join(cuts)}]" for cuts in unique_cut_patterns)
        print(f"Board length: {stock_length}\" - Quantity: {total_quantity} - Cuts: [{cut_patterns_str}]")

def print_detailed_solution(result: Dict):
    """Print solution with theoretical minimums and gaps"""
    print(f"\nOptimal Board Cutting Solution:")
    print(f"Status: {result['status']}")
    
    theory = result['theoretical_minimums']
    print("\nTheoretical Minimums:")
    print(f"Total length needed: {theory['total_length_needed']:.2f} inches")
    print(f"Minimum theoretical cost: ${theory['min_theoretical_cost']:.2f}")
    print(f"Minimum theoretical waste: {theory['min_theoretical_waste']:.2f} inches")
    print(f"Most efficient board: {theory['price_efficiency']['board_length']:.1f} inches "
          f"(${theory['price_efficiency']['price_per_inch']:.3f}/inch)")
    
    print(f"\nActual Solution:")
    print(f"Total Cost: ${result['total_cost']:.2f} "
          f"({result['solution_quality']['cost_gap']:.1f}% above theoretical minimum)")
    print(f"Total Waste: {result['total_waste']:.2f} inches "
          f"({result['solution_quality']['waste_gap']:.1f}% above theoretical minimum)")
    print(f"Column Generation Iterations: {result['iterations']}")
    
    print("\nCutting Patterns:")
    for i, pattern in enumerate(result['cutting_patterns'], 1):
        print(f"\nPattern {i} (Use {pattern['times_used']} times):")
        print(f"  Stock length: {pattern['stock_length_feet']:.1f}' ({pattern['stock_length']:.1f}\")")
        print("  Cuts:")
        for detail in pattern['cut_details']:
            print(f"    {detail['count']}x {detail['length']}\" - {detail['description']}")
        waste = pattern['stock_length'] - sum(pattern['cuts'])
        print(f"  Waste per board: {waste:.1f}\" ({waste/12:.2f}')")

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Cutting Stock Optimizer')
    parser.add_argument('prices_file', type=str, help='CSV file containing stock lengths and prices')
    parser.add_argument('parts_file', type=str, help='CSV file containing required parts and quantities')
    
    # Parse arguments
    args = parser.parse_args()

    try:
        # Create solver instance
        solver = CuttingStockSolver(args.prices_file, args.parts_file)
        
        # Get optimal solution
        result = solver.optimize()
        
        # Print both formats
        print("Simple Output Format:")
        print_simple_solution(result)

        print("\nSimple Output Format Collapsed:")
        print_collapsed_solution(result)
        
        print("\nDetailed Output Format:")
        print_detailed_solution(result)

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {str(e)}")
    except pd.errors.EmptyDataError:
        print("Error: One or both CSV files are empty")
    except pd.errors.ParserError:
        print("Error: Invalid CSV file format")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()