import pygame
import neat
import os
import random

pygame.init()


class Paddle(pygame.sprite.Sprite):
    def __init__(self, screen, this_instance, net, genome):
        super().__init__()

        self.this_instance = this_instance

        self.net = net
        self.genome = genome

        self.screen = screen

        self.key_state = None

        self.image = pygame.Surface([25, 200])
        self.image.fill((20, 60, 150))

        self.rect = self.image.get_rect()

        self.xPos = screen.get_width() - 60
        self.yPos = (screen.get_height() / 2) - (self.image.get_height() / 2)
        self.velocity = 0

    def update(self):

        self.rect = pygame.rect.Rect([self.xPos, self.yPos, 25, 200])

        out = self.net.activate((self.yPos, self.this_instance.ball.xPos, self.this_instance.ball.yPos))

        if out[0] > 0.5:
            self.velocity = 3
        elif out[0] < 0.5:
            self.velocity = -3
        elif out[1] > 0.5:
            self.velocity = 0

        self.yPos += self.velocity * -1

        if self.yPos <= 0:
            self.yPos = 0
        elif self.yPos >= self.screen.get_height() - self.image.get_height():
            self.yPos = self.screen.get_height() - self.image.get_height()

        self.draw()
        pygame.event.pump()

    def draw(self):
        self.screen.blit(self.image, (self.xPos, self.yPos))

    def kill_genome(self):
        self.genome.fitness -= 1
        instances.pop(instances.index(self.this_instance))


class Ball(pygame.sprite.Sprite):
    def __init__(self, screen, paddle):
        super().__init__()

        self.screen = screen
        self.paddle = paddle

        self.image = pygame.Surface([10, 10])
        self.image.fill((150, 0, 150))

        self.rect = self.image.get_rect()

        self.xPos = screen.get_width() / 2
        self.yPos = screen.get_height() / 2
        self.velocity = [3, -3]

    def update(self):

        self.rect = pygame.rect.Rect([self.xPos, self.yPos, 10, 10])

        self.xPos += self.velocity[0]
        self.yPos += self.velocity[1]

        if self.xPos >= self.screen.get_width() - self.image.get_width():
            self.paddle.kill_genome()

        if pygame.sprite.collide_mask(self, self.paddle):
            if self.xPos <= self.paddle.xPos:
                self.velocity[0] *= -1
                # self.velocity[1] = random.randint(-8, 8)
                self.xPos -= 7
            elif self.yPos <= self.paddle.yPos + self.paddle.rect[3]:
                self.velocity[1] *= -1
                self.yPos += 7
            elif self.yPos + self.rect[3] >= self.paddle.yPos:
                self.velocity[1] *= -1
                self.yPos -= 7

        if self.yPos >= self.screen.get_height() - self.image.get_height():
            self.velocity[1] *= -1
        if self.yPos <= 0:
            self.velocity[1] *= -1
        if self.xPos <= 100:
            self.velocity[0] *= -1

        self.draw()

    def draw(self):
        self.screen.blit(self.image, (self.xPos, self.yPos))


class Instance:
    def __init__(self, screen, net, genome):
        self.paddle = Paddle(screen, self, net, genome)
        self.ball = Ball(screen, self.paddle)

    def update(self):
        self.paddle.update()
        self.ball.update()


screen = pygame.display.set_mode((1100, 800))
clock = pygame.time.Clock()

wall = pygame.Rect((50, 0), (20, screen.get_height()))

instances = []


def eval_genomes(genomes, config):
    global instances

    instances = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        instances.append(Instance(screen, net, genome))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        screen.fill((0, 0, 0))

        pygame.draw.rect(screen, (255, 255, 255), (80, 0, 20, screen.get_height()))

        for i in instances:
            i.update()

        if len(instances) == 0:
            return

        pygame.display.update()

        clock.tick(60)


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 500)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
