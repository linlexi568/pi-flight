# Makefile for π-Flight Controller C Implementation

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11
TARGET = example
OBJS = pi_flight_controller.o example.o

# Default target: build the example program
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)
	@echo "Build successful! Run with: ./$(TARGET)"

# Compile controller implementation
pi_flight_controller.o: pi_flight_controller.c pi_flight_controller.h
	$(CC) $(CFLAGS) -c pi_flight_controller.c

# Compile example program
example.o: example.c pi_flight_controller.h
	$(CC) $(CFLAGS) -c example.c

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET) $(TARGET).exe

# Run the example
run: $(TARGET)
	./$(TARGET)

# Static library (optional, for integration into other projects)
libpiflight.a: pi_flight_controller.o
	ar rcs libpiflight.a pi_flight_controller.o
	@echo "Static library created: libpiflight.a"

.PHONY: all clean run
