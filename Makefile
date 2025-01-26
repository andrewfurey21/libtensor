# TODO: more env vars, make install

DEBUG=1

ALIB=libtensor.a
OBJDIR=./obj/
SRCDIR=./src/

CC=gcc
AR=ar
ARFLAGS=rcs
OPTS=-O3
LDFLAGS= -lm
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall

EXAMPLE=./examples/mnist.c
EXEC=example
EXECDIR=./build

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

OBJ=graph.o op.o optimizers.o tuple.o tensor.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = Makefile include/tensor.h

all: $(EXECDIR) $(OBJDIR) $(ALIB) $(EXEC) $(DEPS)

$(EXEC): $(ALIB) $(OBJDIR) $(EXAMPLE)
	$(CC) $(COMMON) $(CFLAGS) $(EXAMPLE) -o $(EXECDIR)/$@ $(ALIB) $(LDFLAGS)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(OBJDIR)%.o: $(SRCDIR)%.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	@mkdir -p $@

$(EXECDIR):
	@mkdir -p $@

.PHONY: clean
clean:
	@rm -rf $(ALIB) $(OBJDIR) $(EXECDIR)
