#include "ThermalStrip.h"

const uint8_t ESC = 27;
const uint8_t DC2 = 18;

const uint8_t MODE_BOLD          = 1 << 3;
const uint8_t MODE_DOUBLE_HEIGHT = 1 << 4;
const uint8_t MODE_DOUBLE_WIDTH  = 1 << 5;

const uint8_t LINE_GAP_DOTS = 6;  // paper fed between text lines

// ---------------------------------------------------------------------------
// ThermalPrinter
// ---------------------------------------------------------------------------
void ThermalPrinter::begin(Uart &port, uint32_t baud) {
  port_ = &port;
  port_->begin(baud);
  readyAt_ = micros() + 500000;  // the printer needs a moment after power-up

  push(ESC, '@');                // reset
  pause(50000);
  push(ESC, '7', 11); push(120, 40);  // heating: dots at once, heat time, heat interval
  push(DC2, '#', (2 << 5) | 10);      // print density and break time
}

void ThermalPrinter::update() {
  while (count_ > 0) {
    if ((int32_t)(micros() - readyAt_) < 0) return;
    if (port_->availableForWrite() <= 0) return;

    uint16_t index = tail_;
    port_->write(queue_[index]);
    tail_ = (tail_ + 1) % QUEUE_SIZE;
    count_--;

    if (pauseCount_ > 0 && pauses_[pauseTail_].after == index) {
      readyAt_ = micros() + pauses_[pauseTail_].us;
      pauseTail_ = (pauseTail_ + 1) % MAX_PAUSES;
      pauseCount_--;
    }
  }
}

bool ThermalPrinter::idle() const {
  return count_ == 0 && (int32_t)(micros() - readyAt_) >= 0;
}

bool ThermalPrinter::hasSpace(uint16_t bytes) const {
  return QUEUE_SIZE - count_ >= bytes && pauseCount_ < MAX_PAUSES;
}

void ThermalPrinter::push(uint8_t b) {
  queue_[head_] = b;
  head_ = (head_ + 1) % QUEUE_SIZE;
  count_++;
}

// Wait `us` after the byte queued last has been sent
void ThermalPrinter::pause(uint32_t us) {
  if (count_ == 0) {
    readyAt_ = micros() + us;
    return;
  }
  uint16_t after = (head_ + QUEUE_SIZE - 1) % QUEUE_SIZE;
  uint8_t last = (pauseHead_ + MAX_PAUSES - 1) % MAX_PAUSES;
  if (pauseCount_ > 0 && pauses_[last].after == after) {
    pauses_[last].us += us;
    return;
  }
  pauses_[pauseHead_] = { after, us };
  pauseHead_ = (pauseHead_ + 1) % MAX_PAUSES;
  pauseCount_++;
}

bool ThermalPrinter::text(const char *s) {
  if (!hasSpace(strlen(s))) return false;
  for (; *s; s++) {
    push(*s);
    if (*s == '\n') pause(charHeight_ * dotPrintUs + LINE_GAP_DOTS * dotFeedUs);
  }
  return true;
}

void ThermalPrinter::style(bool bold, bool large) {
  uint8_t mode = 0;
  if (bold) mode |= MODE_BOLD;
  if (large) mode |= MODE_DOUBLE_HEIGHT | MODE_DOUBLE_WIDTH;
  charHeight_ = large ? 48 : 24;
  push(ESC, '!', mode);
}

void ThermalPrinter::align(char where) {
  push(ESC, 'a', where == 'C' ? 1 : where == 'R' ? 2 : 0);
}

void ThermalPrinter::feed(uint8_t lines) {
  push(ESC, 'd', lines);
  pause(lines * (charHeight_ + LINE_GAP_DOTS) * dotFeedUs);
}

bool ThermalPrinter::row(const uint8_t *dots) {
  if (!hasSpace(4 + ROW_BYTES)) return false;
  push(DC2, '*', 1); push(ROW_BYTES);  // bitmap: 1 row, ROW_BYTES wide
  for (uint8_t i = 0; i < ROW_BYTES; i++) push(dots[i]);
  pause(dotPrintUs);
  return true;
}

// ---------------------------------------------------------------------------
// StripChart
// ---------------------------------------------------------------------------
static void setDot(uint8_t *row, int16_t x) {
  if (x < 0 || x >= ThermalPrinter::WIDTH_DOTS) return;
  row[x >> 3] |= 0x80 >> (x & 7);
}

static void fillSpan(uint8_t *row, int16_t a, int16_t b) {
  if (a > b) { int16_t t = a; a = b; b = t; }
  for (int16_t x = a; x <= b; x++) setDot(row, x);
}

void StripChart::addLane(int16_t left, int16_t right, float min, float max,
                         uint8_t width, uint8_t dash, bool soft) {
  if (laneCount_ >= MAX_LANES) return;
  lanes_[laneCount_++] = { left, right, min, max, width, dash, soft, -1 };
}

void StripChart::addGuide(uint8_t lane, float value) {
  if (guideCount_ >= MAX_GUIDES || lane >= laneCount_) return;
  guides_[guideCount_++] = position(lane, value);
}

int16_t StripChart::position(uint8_t lane, float value) const {
  const Lane &l = lanes_[lane];
  float f = (value - l.min) / (l.max - l.min);
  if (l.soft) f = 0.5f + 0.5f * tanhf((f - 0.5f) * 2);
  f = constrain(f, 0.0f, 1.0f);
  return l.left + lroundf(f * (l.right - l.left));
}

void StripChart::start() {
  samples_ = 0;
  for (uint8_t i = 0; i < laneCount_; i++) lanes_[i].lastX = -1;
}

bool StripChart::add(const float *values, uint8_t marks) {
  if (!printer_.hasSpace(rowsPerSample_ * (4 + ThermalPrinter::ROW_BYTES))) return false;

  // Where each trace lands this sample
  int16_t x[MAX_LANES];
  for (uint8_t i = 0; i < laneCount_; i++) {
    x[i] = isnan(values[i]) ? -1 : position(i, values[i]);
  }

  bool onGrid = grid_ > 0 && samples_ % grid_ == 0;
  bool onTick = tick_ > 0 && samples_ % tick_ == 0;

  for (uint8_t r = 0; r < rowsPerSample_; r++) {
    uint8_t row[ThermalPrinter::ROW_BYTES] = { 0 };
    uint32_t rowNumber = samples_ * rowsPerSample_ + r;

    for (uint8_t i = 0; i < laneCount_; i++) {
      const Lane &lane = lanes_[i];

      // Guides: dotted lane edges, solid zero line, ticks and grid lines for time
      if (rowNumber % 2 == 0) {
        setDot(row, lane.left);
        setDot(row, lane.right);
      }
      if (lane.min < 0 && lane.max > 0) setDot(row, position(i, 0));
      if (r == 0 && onGrid) {
        for (int16_t d = lane.left; d <= lane.right; d += 3) setDot(row, d);
      } else if (r == 0 && onTick) {
        fillSpan(row, lane.left, lane.left + 4);
        fillSpan(row, lane.right - 4, lane.right);
      }

      if (marks & (1 << i)) fillSpan(row, lane.right + 3, lane.right + 8);

      // Trace, joined to the previous point; dashed traces skip every other stretch of rows
      if (x[i] < 0) continue;
      if (lane.dash > 0 && (rowNumber / lane.dash) % 2 == 1) continue;
      int16_t from = lane.lastX < 0 ? x[i] : lane.lastX;
      int16_t a = from + (x[i] - from) * r / rowsPerSample_;
      int16_t b = from + (x[i] - from) * (r + 1) / rowsPerSample_;
      fillSpan(row, min(a, b) - (lane.width - 1) / 2, max(a, b) + lane.width / 2);
    }

    if (rowNumber % 6 == 0) {
      for (uint8_t g = 0; g < guideCount_; g++) setDot(row, guides_[g]);
    }
    printer_.row(row);
  }

  for (uint8_t i = 0; i < laneCount_; i++) lanes_[i].lastX = x[i];
  samples_++;
  return true;
}

bool StripChart::gap() {
  for (uint8_t i = 0; i < laneCount_; i++) lanes_[i].lastX = -1;

  const uint8_t ROWS = 2;
  if (!printer_.hasSpace(ROWS * (4 + ThermalPrinter::ROW_BYTES))) return false;
  uint8_t row[ThermalPrinter::ROW_BYTES] = { 0 };
  for (uint8_t i = 0; i < laneCount_; i++) {
    for (int16_t x = lanes_[i].left; x <= lanes_[i].right; x++) {
      if ((x - lanes_[i].left) % 8 < 4) setDot(row, x);  // dashes, unlike the dotted grid lines
    }
  }
  for (uint8_t r = 0; r < ROWS; r++) printer_.row(row);
  return true;
}
