import pygame
from checkers.constants import *
from gomoku import Gomoku
import pygame.mixer

pygame.init()

FPS = 60

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Gomuko')

# button_font = pygame.font.Font(None,74)
# button_text = button_font.render('Play Game',True,WHITE )
# button_rect = button_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))

start_img = pygame.image.load('images/start_btn.png').convert_alpha()
exit_img = pygame.image.load('images/exit_btn.png').convert_alpha()
background_image = pygame.image.load('images/oneb.jpg')
1
class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)

    def draw(self):
        window.blit(self.image, (self.rect.x, self.rect.y))

        


# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED= (255, 0, 0)
GREEN= (0, 255, 0)
# BLUE =  (173, 216, 230)
MARON = (128, 0, 0)
class TextButton:
    def __init__(self, x, y, w, h, text, base_color=MARON, hover_color=BLUE, press_color=WHITE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font_size = 36
        self.font = pygame.font.Font(None, self.font_size)
        self.base_color = base_color
        self.hover_color = hover_color
        self.press_color = press_color
        self.current_color = base_color
        self.update_text_surface()
        self.pressed = False

    def update_text_surface(self):
        # Render the text and get its rectangle
        self.text_surf = self.font.render(self.text, True, WHITE)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def draw(self, surface):
        # Draw the button background
        pygame.draw.rect(surface, self.current_color, self.rect)
        pygame.draw.rect(surface, GREEN, self.rect, 2)  # Border around the button
        # Draw the text on top of the button
        surface.blit(self.text_surf, self.text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                self.current_color = self.hover_color
            else:
                self.current_color = self.base_color
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.current_color = self.press_color
            self.pressed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed and self.rect.collidepoint(event.pos):
                self.current_color = self.hover_color
                self.pressed = False
                return True
            else:
                self.current_color = self.base_color
                self.pressed = False
        return False
        


start_btn = Button(WIDTH // 2 - start_img.get_width() // 2, HEIGHT // 2 - start_img.get_height() // 2, start_img)
exit_btn = Button(WIDTH // 2 - exit_img.get_width() // 2, HEIGHT // 2 - exit_img.get_height() // 2, exit_img)
new_game_btn = TextButton(300, 800, 200, 60, "New Game")


# new_game_btn = Button()

def draw_menu():
    #window.fill(BackGroundColor)
    # Draw the background image
    window.blit(background_image, (0, 0))
    start_btn.draw()
    pygame.display.flip()


def draw_game(game):
    game.draw_board(window)
    pygame.display.flip()

def draw_popup(surface, message):
    # Popup size and position
    popup_width, popup_height = 400, 200
    popup_x = (surface.get_width() - popup_width) // 2
    popup_y = (surface.get_height() - popup_height) // 2
    
    # Draw the popup background
    pygame.draw.rect(surface, BROWN, (popup_x, popup_y, popup_width, popup_height))
    
    # Draw the border of the popup
    pygame.draw.rect(surface, BLACK, (popup_x, popup_y, popup_width, popup_height), 2)
    
    # Render the message text
    font = pygame.font.Font(None, 100)
    text_surface = font.render(message, True, WHITE)
    text_rect = text_surface.get_rect(center=(popup_x + popup_width // 2, popup_y + popup_height // 2))
    surface.blit(text_surface, text_rect)