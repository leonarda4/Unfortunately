#pragma once

#include <Arduino.h>

// ---------------------------------------------------------------------------
// Non-blocking driver for the Adafruit mini thermal printer (CSN-A2, TTL).
//
// Everything is queued and sent from update() a few bytes at a time, paced so
// the printer's small input buffer never overflows. The caller's loop keeps
// running while the printer works.
// ---------------------------------------------------------------------------
class ThermalPrinter {
 public:
  static const uint16_t WIDTH_DOTS = 384;
  static const uint8_t  ROW_BYTES  = WIDTH_DOTS / 8;

  // Rough time the printer needs per dot line. Same conservative values as the
  // Adafruit library; lower them if printing turns out to be faster.
  uint32_t dotPrintUs = 30000;
  uint32_t dotFeedUs  = 2100;

  void begin(Uart &port, uint32_t baud);
  void update();           // call every loop(): sends what the printer can take now
  bool idle() const;       // nothing queued and the printer should be done
  uint16_t queued() const { return count_; }
  bool hasSpace(uint16_t bytes) const;

  // Text. Characters are queued as they are, "\n" prints the line.
  bool text(const char *s);
  void style(bool bold, bool large);  // large = double width and height
  void align(char where);             // 'L', 'C' or 'R'
  void feed(uint8_t lines);

  // One 384-dot row of graphics, bit 7 of byte 0 is the leftmost dot.
  // Returns false (and queues nothing) if the queue is full.
  bool row(const uint8_t *dots);

 private:
  static const uint16_t QUEUE_SIZE = 4096;
  static const uint8_t  MAX_PAUSES = 128;
  struct Pause {
    uint16_t after;  // queue index of the byte after which to wait
    uint32_t us;
  };

  void push(uint8_t b);
  void push(uint8_t a, uint8_t b) { push(a); push(b); }
  void push(uint8_t a, uint8_t b, uint8_t c) { push(a); push(b); push(c); }
  void pause(uint32_t us);

  Uart *port_ = nullptr;
  uint8_t queue_[QUEUE_SIZE];
  uint16_t head_ = 0, tail_ = 0, count_ = 0;
  Pause pauses_[MAX_PAUSES];
  uint8_t pauseHead_ = 0, pauseTail_ = 0, pauseCount_ = 0;
  uint32_t readyAt_ = 0;  // micros() before which nothing may be sent
  uint8_t charHeight_ = 24;
};

// ---------------------------------------------------------------------------
// Strip chart: each sample becomes a few dot rows, so time runs down the paper
// and each signal gets a lane across it.
// ---------------------------------------------------------------------------
class StripChart {
 public:
  static const uint8_t MAX_LANES  = 4;
  static const uint8_t MAX_GUIDES = 8;

  explicit StripChart(ThermalPrinter &printer, uint8_t rowsPerSample = 2)
      : printer_(printer), rowsPerSample_(rowsPerSample) {}

  // Dot range across the paper and the value range it shows. Lanes may overlap
  // to draw several traces on one scale. `width` is the trace thickness in dots,
  // `dash` the length of its dashes in dot rows (0 = solid). With `soft`, big
  // values are squeezed instead of cut off: the trace is 3/4 of the way from
  // the centre to the edge at min / max and only approaches the edge beyond.
  void addLane(int16_t left, int16_t right, float min, float max,
               uint8_t width = 3, uint8_t dash = 0, bool soft = false);
  // Faint vertical line where `value` lies on lane `lane`, to read the scale
  void addGuide(uint8_t lane, float value);
  // Dot position of `value` on lane `lane`
  int16_t position(uint8_t lane, float value) const;
  // Ticks on the lane edges every `tickSamples`, a dotted line across every `gridSamples`
  void setGrid(uint16_t tickSamples, uint16_t gridSamples) { tick_ = tickSamples; grid_ = gridSamples; }
  void start();  // begin a new chart: traces and time grid restart

  // One value per lane (NAN leaves a gap). Bit i of `marks` draws a marker next
  // to lane i. Returns false if the printer queue is full and the sample was dropped.
  bool add(const float *values, uint8_t marks = 0);

  // Dashed line across the lanes where the recording paused; traces restart
  // after it instead of joining across. Returns false if the queue is full.
  bool gap();

 private:
  struct Lane {
    int16_t left, right;
    float min, max;
    uint8_t width, dash;
    bool soft;
    int16_t lastX;  // -1 = no previous point
  };

  ThermalPrinter &printer_;
  uint8_t rowsPerSample_;
  Lane lanes_[MAX_LANES];
  uint8_t laneCount_ = 0;
  int16_t guides_[MAX_GUIDES];
  uint8_t guideCount_ = 0;
  uint16_t tick_ = 10, grid_ = 100;
  uint32_t samples_ = 0;
};
