class KakuroSolver:
    def __init__(self, board, row_constraints, col_constraints):
        self.board = board
        self.row_constraints = row_constraints
        self.col_constraints = col_constraints
        self.size = len(board)
        self.empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if board[i][j] == -1]

    def solve(self, index=0):
        if index == len(self.empty_cells):
            return True

        row, col = self.empty_cells[index]
        for num in range(1, 10):
            if self._is_valid(row, col, num):
                self.board[row][col] = num
                if self.solve(index + 1):
                    return True
                self.board[row][col] = -1

        return False

    def _is_valid(self, row, col, num):
        return self._check_sum_constraint(row, col, num, self.row_constraints, vertical=False) and \
               self._check_sum_constraint(row, col, num, self.col_constraints, vertical=True)

    def _check_sum_constraint(self, x, y, num, constraints, vertical):
        if (x, y) not in constraints:
            return True  # No constraint for this cell

        sum_constraint = constraints[(x, y)]
        current_sum, nums_used, empty_cells = 0, set(), 0

        if vertical:
            start_row = x
            while start_row > 0 and self.board[start_row - 1][y] != -1:
                start_row -= 1

            for r in range(start_row, self.size):
                if self.board[r][y] == -1:
                    empty_cells += 1
                    if empty_cells > 1:
                        break  # Can't have more than one empty cell in a sum block
                elif self.board[r][y] > 0:
                    current_sum += self.board[r][y]
                    if self.board[r][y] in nums_used:
                        return False  # Duplicate number in the sum block
                    nums_used.add(self.board[r][y])

        else:
            start_col = y
            while start_col > 0 and self.board[x][start_col - 1] != -1:
                start_col -= 1

            for c in range(start_col, self.size):
                if self.board[x][c] == -1:
                    empty_cells += 1
                    if empty_cells > 1:
                        break  # Can't have more than one empty cell in a sum block
                elif self.board[x][c] > 0:
                    current_sum += self.board[x][c]
                    if self.board[x][c] in nums_used:
                        return False  # Duplicate number in the sum block
                    nums_used.add(self.board[x][c])

        if empty_cells == 1 and num in nums_used:
            return False  # The only empty cell can't have the same number as another cell in the sum block

        if num in nums_used or current_sum + num != sum_constraint:
            return False  # Violates sum constraint or duplicates a number

        return True

    def print_board(self):
        for row in self.board:
            print(' '.join(str(num) if num != -1 else '_' for num in row))


board = [
    [-1, -1, -1],
    [-1, -1, 22],
    [-1, 7, -1],
]

row_constraints = {
    (1, 0): 30,
}

col_constraints = {
    (0, 2): 23,
}

solver = KakuroSolver(board, row_constraints, col_constraints)
if solver.solve():
    solver.print_board()
else:
    print("No solution exists")
