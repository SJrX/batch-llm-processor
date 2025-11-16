"""
Calculator tools for Anthropic API using @beta_tool decorator
"""

import math
from anthropic import beta_tool


@beta_tool
def add(a: float, b: float) -> str:
    """Add two numbers together.
    
    Args:
        a: The first number to add
        b: The second number to add
        
    Returns:
        The sum of a and b
    """
    return str(a + b)


@beta_tool
def subtract(a: float, b: float) -> str:
    """Subtract the second number from the first number.
    
    Args:
        a: The number to subtract from
        b: The number to subtract
        
    Returns:
        The result of a - b
    """
    return str(a - b)


@beta_tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together.
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
        
    Returns:
        The product of a and b
    """
    return str(a * b)


@beta_tool
def divide(a: float, b: float) -> str:
    """Divide the first number by the second number.
    
    Args:
        a: The dividend (number to divide)
        b: The divisor (number to divide by)
        
    Returns:
        The result of a / b
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return str(a / b)


@beta_tool
def sqrt(x: float) -> str:
    """Calculate the square root of a number.
    
    Args:
        x: The number to calculate the square root of (must be non-negative)
        
    Returns:
        The square root of x
    """
    if x < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return str(math.sqrt(x))


@beta_tool
def power(base: float, exponent: float) -> str:
    """Raise a number to a power (exponentiation).
    
    Args:
        base: The base number
        exponent: The exponent (power)
        
    Returns:
        The result of base raised to the power of exponent
    """
    return str(base ** exponent)


# All tools for easy import
TOOLS = [add, subtract, multiply, divide, sqrt, power]
