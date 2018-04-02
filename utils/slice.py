from typing import List


__all__ = ['fixed_length_slice_with_pad']


def fixed_length_slice_with_pad(x: List, s: int, l: int):
  """Extract a fixed length slice from List `x`, out-of-range components are filled with edge values.

  Example:
    a = list(range(5))  # [0, 1, 2, 3, 4]
    fixed_length_slice_with_pad(a, -1, 7)  # [0, 0, 1, 2, 3, 4, 4]
  """
  lx = len(x)  # length of list `x`
  if s > lx:
    raise ValueError('s({}) exceeds array length ({})'.format(s, lx))
  e = s + l    # end
  if e < 0:
    raise ValueError('e({}) = s({}) + l({}) should be non-negative'.format(e, s, l))
  (sp, s) = (-s, 0) if s < 0 else (0, s)
  (ep, e) = (e - lx, lx) if e > lx else (0, e)
  return [x[0] for _ in range(sp)] + x[slice(s, e)] + [x[-1] for _ in range(ep)]


def main():
  x = list(range(10))
  print(x)
  try:
    print(fixed_length_slice_with_pad(x, -20, 6))
  except ValueError:
    print('Left Error')
  print(fixed_length_slice_with_pad(x, -2, 6))
  print(fixed_length_slice_with_pad(x, 0, 6))
  print(fixed_length_slice_with_pad(x, 2, 6))
  print(fixed_length_slice_with_pad(x, -1, 12))
  print(fixed_length_slice_with_pad(x, 6, 6))
  try:
    print(fixed_length_slice_with_pad(x, 12, 6))
  except ValueError:
    print('Right Error')


if __name__ == '__main__':
  main()
