def parse_value(value):
  # Try to convert to integer
  try:
      return int(value)
  except ValueError:
      pass
  
  # Try to convert to float
  try:
      return float(value)
  except ValueError:
      pass
  
  # Return as string if both conversions fail
  return value