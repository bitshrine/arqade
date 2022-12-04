from games.game import Game, STATE_GAME_OVER
import curses, numpy as np

class Display():
    def __init__(self, game: Game):
        """
        Initialize the display, giving it a Game to render
        """
        self.game = game

        self.stdscr = curses.initscr()

        win = curses.newwin(game.render_state.shape[0] + 1, game.render_state.shape[1] + 1, 0, 0)
        curses.curs_set(0)
        win.timeout(0)

        curses.start_color()
        self._define_colors()

        self.win = win
        self.draw()

    def draw(self):
        """
        Draw the game's render state onto the display
        """
        if (self.game.game_state == STATE_GAME_OVER): 
            self.kill()
        else:
            self.win.clear()
            self.win.border(0)
            self.win.addstr(0, 2, ' Score : ' + str(self.game.score) + ' ')
            for px in np.transpose(np.nonzero(self.game.draw())):
                #self.win.addch(px[0], px[1], '█')
                char, c = self.game._get_pixel(px)
                #self.win.addstr(px[0], px[1], '█', curses.color_pair(self.colors[self.game.render_state[px[0]][px[1]]]))
                self.win.addstr(px[0], px[1], char, curses.color_pair(self.colors[c]))
            self.win.getch()

    def kill(self):
        """
        Shutdown the display
        """
        curses.curs_set(1)
        curses.endwin()


    def _define_colors(self):
        """
        Define the color of cells which have non-zero value
        """
        self.colors = {
            curses.COLOR_BLACK: 1,
            curses.COLOR_RED: 2,
            curses.COLOR_GREEN: 3,
            curses.COLOR_YELLOW: 4,
            curses.COLOR_BLUE: 5,
            curses.COLOR_MAGENTA: 6,
            curses.COLOR_CYAN: 7,
            curses.COLOR_WHITE: 8
        }
        for (col, idx) in self.colors.items():
            curses.init_pair(idx, col, curses.COLOR_BLACK)